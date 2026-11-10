"""Run the visual, textual, structural, and fuzzy agents with a TransE checkpoint.

Example from the repository root:
    python -m vlkge.mas.run_fuzzy_transe --config vlkge/configs/wn9_img/transe_clip.yaml \
        --checkpoint runs/wn9_img/transe_multimodal/seed42.pt \
        --visual-range MIN MAX --textual-range MIN MAX --structural-range MIN MAX \
        --head n11669921 --relation _hyponym --tail n11992806

The score ranges must be estimated on training/validation data. The modality
scores are post-hoc evidence from the standard fused TransE checkpoint.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from vlkge import helpers, utils
from vlkge.dataloader import KnowledgeGraphDataLoader
from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.evidence_adapter import EvidenceAdapter
from vlkge.mas.fuzzy_aggregator import FuzzyEvidenceAggregator
from vlkge.scripts.train import REPO_ROOT, parse_args as parse_training_args


def load_transe_fuzzy_environment(config_path, checkpoint_path, score_ranges,
                                  device, seed=42, decision_threshold=None):
    """Rebuild the configured TransE, load its checkpoint, and create agents."""
    trainer_args = ["--config", str(config_path)]
    if device.type == "cpu":
        trainer_args.append("--no_cuda")
    config = parse_training_args(trainer_args)
    if config.model != "TransE":
        raise ValueError("The fuzzy TransE runner requires a TransE training config.")
    if not (config.use_structural and config.use_visual and config.use_textual):
        raise ValueError(
            "The fuzzy TransE runner requires structural, visual, and textual modalities."
        )
    if config.fusion_mode == "concat":
        raise ValueError(
            "Post-hoc modality evidence currently supports TransE fusion modes "
            "other than 'concat'."
        )
    if config.dataset != "wn9_img":
        raise ValueError("This runner currently supports the WN9-IMG dataset only.")

    loader = KnowledgeGraphDataLoader(
        data_path=config.data_path,
        dataset_name=config.dataset,
        exclude_relations=config.exclude_relations,
        exclude_relations_eval=config.exclude_relations_eval,
        add_inverse_relations=config.add_inverse_relations,
        use_per_relation_candidates=config.use_per_relation_candidates,
        artist2artist_relations=config.artist2artist_relations,
        bidirectional_eval=config.bidirectional_eval,
    )
    entity_to_id, relation_to_id = loader.get_entities_and_relations()
    features = utils.load_features(config, entity_to_id, relation_to_id)
    visual, textual, relation, visual_mapping, textual_mapping = features
    if set(visual_mapping) != set(entity_to_id.values()):
        raise ValueError("WN9-IMG fuzzy inference requires visual features for every entity.")
    if set(textual_mapping) != set(entity_to_id.values()):
        raise ValueError("WN9-IMG fuzzy inference requires textual features for every entity.")

    model, _, _ = helpers.get_model(
        model_name="TransE",
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        visual_features=visual,
        textual_features=textual,
        relation_features=relation,
        visual_entity_to_index=visual_mapping,
        textual_entity_to_index=textual_mapping,
        embedding_dim=config.embedding_dim,
        fusion_mode=config.fusion_mode,
        use_structural=config.use_structural,
        use_visual=config.use_visual,
        use_textual=config.use_textual,
        freeze_visual=config.freeze_visual,
        freeze_textual=config.freeze_textual,
        visual_proj=config.visual_proj,
        textual_proj=config.textual_proj,
        shared_projection=config.shared_projection,
        inductive=config.inductive,
        modality_asymmetry=config.modality_asymmetry,
        normalize_before_fusion=config.normalize_before_fusion,
        p_norm=config.p_norm,
        normalize_relations=config.normalize_relations,
        margin=config.margin,
        lr=config.lr,
        use_scheduler=False,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a model_state_dict.")
    if checkpoint.get("model_name") == "TransEFuzzy":
        raise ValueError(
            "This checkpoint was trained as TransEFuzzy. It already includes "
            "fuzzy fusion in its TransE scoring function."
        )
    if checkpoint.get("model_name", "TransE") != "TransE":
        raise ValueError("The supplied checkpoint is not a TransE checkpoint.")
    if checkpoint.get("entity_to_id", entity_to_id) != entity_to_id:
        raise ValueError("Dataset entity IDs differ from the checkpoint.")
    if checkpoint.get("relation_to_id", relation_to_id) != relation_to_id:
        raise ValueError("Dataset relation IDs differ from the checkpoint.")
    training_config = checkpoint.get("training_config") or {}
    if training_config.get("fuzzy_fusion", False):
        raise ValueError(
            "This checkpoint already uses train-time fuzzy TransE fusion. "
            "Do not apply the post-hoc fuzzy MAS aggregator to it again."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()

    adapter = EvidenceAdapter(model, entity_to_id, relation_to_id, device)
    aggregator = FuzzyEvidenceAggregator(score_ranges=score_ranges)
    environment = KnowledgeGraphEnvironment(
        visual_evidence_model=adapter.score_visual,
        textual_evidence_model=adapter.score_textual,
        structural_evidence_model=adapter.score_structural,
        seed=seed,
        decision_threshold=decision_threshold,
        fuzzy_aggregator=aggregator,
    )
    return environment, model, loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--visual-range", type=float, nargs=2, required=True,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--textual-range", type=float, nargs=2, required=True,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--structural-range", type=float, nargs=2, required=True,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    for name in ("visual_range", "textual_range", "structural_range"):
        minimum, maximum = getattr(args, name)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            parser.error(f"--{name.replace('_', '-')} must be finite and increasing.")
    if args.decision_threshold is not None and not math.isfinite(args.decision_threshold):
        parser.error("--decision-threshold must be finite.")
    return args


def main(argv=None):
    args = parse_args(argv)
    utils.set_seed(args.seed)
    device = torch.device("cpu") if args.no_cuda else utils.set_gpu()
    checkpoint_path = utils.validate_path(
        utils.expand_path(args.checkpoint), "TransE checkpoint"
    )
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO_ROOT / config_path
    config_path = utils.validate_path(str(config_path), "Training config")
    score_ranges = {
        "visual": tuple(args.visual_range),
        "textual": tuple(args.textual_range),
        "structural": tuple(args.structural_range),
    }
    environment, _, _ = load_transe_fuzzy_environment(
        config_path, checkpoint_path, score_ranges, device,
        seed=args.seed, decision_threshold=args.decision_threshold,
    )
    candidate = {"head": args.head, "relation": args.relation, "tail": args.tail}
    environment.publish_candidate(candidate)
    environment.step()
    print(json.dumps({"candidate": candidate, "messages": environment.messages},
                     indent=2, allow_nan=False))
    return environment.decisor_agent.last_message


if __name__ == "__main__":
    main()
