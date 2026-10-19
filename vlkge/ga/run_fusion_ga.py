"""Run the train-once, freeze, evolve-alpha experiment on WN9-IMG.

First train with ``python -m vlkge.scripts.train --config
vlkge/configs/wn9_img/mas_gatranse_clip.yaml``. Then run this module with
--config pointing to that same file and --checkpoint pointing to seed42.pt.
The checkpoint must contain training_config and ID mappings, which the
updated trainer saves. This prevents silent scoring with incompatible IDs
or model options. No relation training occurs during this command.

Validation selects all GA/baseline weights. --evaluate-test explicitly adds
final test metrics only after that selection. --weights can reuse a previous
result JSON to evaluate its selected alpha without repeating the GA search.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vlkge import helpers, utils
from vlkge.dataloader import KnowledgeGraphDataLoader
from vlkge.ga.evaluation import build_score_cache, compare_searches
from vlkge.ga.modality_fusion_ga import GAConfig, run_ga


REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    """Fingerprint files in chunks for checkpoint/dataset provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_experiment(config_path, checkpoint_path, device):
    """Rebuild the trained scorer, verify metadata, load and freeze all weights.

    The first version supports dense WN9-IMG and all-entity candidates. A
    checkpoint trained with another fusion mode, feature policy, mapping, or
    dataset is rejected. Feature values are restored from state_dict; feature
    files reconstruct shapes and the non-buffer entity-to-feature lookups.
    """
    path = Path(config_path).resolve()
    config = utils.resolve_config_paths(
        utils.load_yaml_config(path), base_dir=path.parent, repo_root=REPO_ROOT
    )
    if config.get("dataset") != "wn9_img" or config.get("model") != "MASGATransE":
        raise ValueError("Use a wn9_img / MASGATransE training configuration.")
    expected_flags = {
        "use_structural": False, "use_visual": True, "use_textual": True,
        "use_relation_features": False, "freeze_visual": True, "freeze_textual": True,
        "visual_proj": False, "textual_proj": False, "inductive": False,
        "modality_asymmetry": False, "use_per_relation_candidates": False,
    }
    for key, expected in expected_flags.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"This experiment requires {key}={expected}.")
        config.setdefault(key, expected)
    if config.get("shared_projection"):
        raise ValueError("Shared projection adapters are outside this initial experiment.")
    if config.get("fusion_mode", "average") != "average":
        raise ValueError("Use fusion_mode='average'; alpha combines modality scores.")
    for key in ("normalize_visual", "normalize_textual", "normalize_relation"):
        config.setdefault(key, False)
    config.setdefault("bidirectional_eval", True)
    config.setdefault("top_k", [1, 3, 10])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "MASGATransE":
        raise ValueError("Checkpoint must come from MASGATransE training.")
    trained = checkpoint.get("training_config")
    if not isinstance(trained, dict):
        raise ValueError("Checkpoint lacks training_config; train once using the updated trainer.")
    signature = inspect.signature(helpers.get_model).parameters
    scoring_keys = (
        "embedding_dim", "fusion_mode", "use_structural", "use_visual", "use_textual",
        "freeze_visual", "freeze_textual", "visual_proj", "textual_proj", "shared_projection",
        "inductive", "modality_asymmetry", "normalize_before_fusion", "p_norm",
        "normalize_relations", "margin",
    )
    for key in scoring_keys:
        default = signature[key].default
        if config.get(key, default) != trained.get(key, default):
            raise ValueError(f"Configuration differs from training for {key}.")
    for key in ("normalize_visual", "normalize_textual", "normalize_relation"):
        if config[key] != trained.get(key, False):
            raise ValueError(f"Configuration differs from training for {key}.")
    if trained.get("_data_sha256") != file_sha256(config["data_path"]):
        raise ValueError("Dataset file differs from training or its fingerprint is missing.")

    loader = KnowledgeGraphDataLoader(
        data_path=config["data_path"], dataset_name="wn9_img",
        exclude_relations=config.get("exclude_relations"),
        exclude_relations_eval=config.get("exclude_relations_eval"),
        add_inverse_relations=config.get("add_inverse_relations"),
        bidirectional_eval=config["bidirectional_eval"],
    )
    entity_to_id, relation_to_id = loader.get_entities_and_relations()
    if entity_to_id != checkpoint.get("entity_to_id") or relation_to_id != checkpoint.get("relation_to_id"):
        raise ValueError("Dataset ID mappings differ from the trained checkpoint.")
    features = utils.load_features(SimpleNamespace(**config), entity_to_id, relation_to_id)
    visual, textual, relation, visual_mapping, textual_mapping = features
    if set(visual_mapping) != set(entity_to_id.values()) or set(textual_mapping) != set(entity_to_id.values()):
        raise ValueError("This WN9-IMG experiment requires both modalities for every entity.")
    # Feature rows are stored in the checkpoint; check their entity ordering too.
    if visual_mapping != checkpoint.get("visual_entity_to_index") or textual_mapping != checkpoint.get("textual_entity_to_index"):
        raise ValueError("Feature row mappings differ from the trained checkpoint.")
    options = {key: config[key] for key in signature if key in config}
    options.update(
        model_name="MASGATransE", num_entities=len(entity_to_id), num_relations=len(relation_to_id),
        visual_features=visual, textual_features=textual, relation_features=relation,
        visual_entity_to_index=visual_mapping, textual_entity_to_index=textual_mapping,
        device=device, use_scheduler=False,
    )
    model, _, _ = helpers.get_model(**options)
    model.load_state_dict(checkpoint["model_state_dict"])
    if model.alpha_tensor.item() != 0.5:
        raise ValueError("Expected a scoring checkpoint trained with alpha fixed at 0.5.")
    model.requires_grad_(False)
    model.eval()
    return model, loader, config


def triples_to_ids(frame, loader):
    """Convert named triples to the exact checkpoint vocabulary order."""
    return np.column_stack((
        frame["head"].map(loader.entity_to_id).to_numpy(dtype=np.int64),
        frame["relation"].map(loader.relation_to_id).to_numpy(dtype=np.int64),
        frame["tail"].map(loader.entity_to_id).to_numpy(dtype=np.int64),
    ))


def parse_args(argv=None):
    """Expose GA settings independently from fixed scoring-model settings.

    Defaults match the agreed 30-individual, tournament/random-reset plan.
    Use --patience 0 to disable GA early stopping. --max-validation-triples
    and --max-test-triples are explicit smoke-run limits, recorded in results;
    omit both for the full experiment. They do not reduce candidate pools.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/wn9_img/mas_gatranse_multimodal/ga_seed42")
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover", choices=("arithmetic", "sbx"), default="arithmetic")
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--mutation-probability", type=float, default=0.2)
    parser.add_argument("--sbx-eta", type=float, default=20)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-batch-size", type=int, default=1024)
    parser.add_argument("--grid-points", type=int, default=None)
    parser.add_argument("--max-validation-triples", type=int, default=None)
    parser.add_argument("--max-test-triples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--weights", help="Reuse fusion_result.json instead of rerunning evolution")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    for name in ("max_validation_triples", "max_test_triples", "grid_points"):
        value = getattr(args, name)
        if value is not None and value < (3 if name == "grid_points" else 1):
            parser.error(f"--{name.replace('_', '-')} is too small.")
    return args


def main(argv=None):
    """Load/freeze, cache validation, evolve, compare, and optionally test.

    Write a JSON report containing checkpoint provenance, operator settings,
    history, baseline metrics, and selected weights. Also save a small weights
    JSON and CSV history; embeddings remain in the original checkpoint.
    Test targets are scored only after every method's alpha has been chosen.
    """
    args = parse_args(argv)
    ga_config = GAConfig(
        population_size=args.population_size, generations=args.generations,
        tournament_size=args.tournament_size, crossover=args.crossover,
        crossover_probability=args.crossover_probability, mutation_probability=args.mutation_probability,
        sbx_eta=args.sbx_eta, elite_size=args.elite_size,
        patience=None if args.patience == 0 else args.patience, seed=args.seed,
    )
    utils.set_seed(args.seed)
    device = utils.set_gpu() if not args.no_cuda else torch.device("cpu")
    model, loader, config = load_experiment(args.config, args.checkpoint, device)
    _, validation, test = loader.split_data()
    if args.max_validation_triples is not None:
        validation = validation.sample(n=min(len(validation), args.max_validation_triples), random_state=args.seed)
    known = loader.data[["hid", "rid", "tid"]].to_numpy(dtype=np.int64)

    def cache(frame, label):
        """Show bounded progress while calculating each split's fixed scores."""
        print(f"Caching {label} scores for {len(frame)} triples...", flush=True)
        return build_score_cache(
            model, triples_to_ids(frame, loader), known,
            bidirectional=config["bidirectional_eval"], candidate_batch_size=args.candidate_batch_size,
            progress=lambda done, total: print(f"  {label}: {done}/{total} queries", flush=True),
        )

    validation_cache = cache(validation, "validation")
    checkpoint_hash = file_sha256(args.checkpoint)
    if args.weights:
        with open(args.weights, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Saved weights belong to a different scoring checkpoint.")
        if previous["config"].get("bidirectional_eval", True) != config["bidirectional_eval"]:
            raise ValueError("Use the original evaluation direction when reusing a search report.")
        previous_arguments = previous["run_arguments"]
        if (previous_arguments.get("max_validation_triples") != args.max_validation_triples
                or previous["validation_triples"] != len(validation)
                or (args.max_validation_triples is not None and previous_arguments["seed"] != args.seed)):
            raise ValueError("Reuse the original validation subset and sampling seed with --weights.")
        ga = previous["ga"]
    else:
        ga = run_ga(
            lambda alpha: validation_cache.metrics(alpha, ks=config["top_k"])["MRR"], ga_config,
            progress=lambda row: print(
                f"Generation {row['generation']}: best MRR={row['best_mrr']:.6f}, "
                f"alpha={row['best_alpha']:.6f}, unique evaluations={row['unique_evaluations']}", flush=True),
        )
    selected = float(ga["best_alpha"])
    methods = compare_searches(validation_cache, ga["unique_evaluations"], args.seed,
                               args.grid_points, ks=config["top_k"])
    methods["ga"] = {"alpha": selected, "validation": validation_cache.metrics(selected, ks=config["top_k"])}
    del validation_cache
    if args.evaluate_test:
        if args.max_test_triples is not None:
            test = test.sample(n=min(len(test), args.max_test_triples), random_state=args.seed)
        test_cache = cache(test, "test")
        for method in methods.values():
            method["test"] = test_cache.metrics(method["alpha"], ks=config["top_k"])
        del test_cache
    model.set_alpha(selected)
    report = {
        "design": "train_once_freeze_evolve_alpha", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_hash, "config": config, "run_arguments": vars(args),
        "filtering": "directional, all-known triples, all-entity candidates",
        "tie_policy": "score descending, entity ID ascending",
        "relation_id_to_name": {str(value): key for key, value in loader.relation_to_id.items()},
        "validation_triples": len(validation), "test_triples": len(test) if args.evaluate_test else None,
        "visual_weight": selected, "textual_weight": 1 - selected,
        "ga": ga, "methods": methods,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fusion_result.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (output / "fusion_weights.json").open("w", encoding="utf-8") as stream:
        json.dump({key: report[key] for key in (
            "checkpoint", "checkpoint_sha256", "visual_weight", "textual_weight"
        )}, stream, indent=2)
    import csv
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ga["history"][0].keys())
        writer.writeheader()
        writer.writerows(ga["history"])
    print(f"Selected visual={selected:.6f}, textual={1-selected:.6f}")
    print(f"Report: {(output / 'fusion_result.json').resolve()}")


if __name__ == "__main__":
    main()
