"""Run GA-weighted agents on every validation (or test) triple.

The actual VisualAgent, TextualAgent, and DecisorAgent execute once for each
observed triple. Save their evidence/decisions to CSV and full messages to
JSONL. Check every agent combined score against direct model inference.

By default, also report filtered link-prediction MRR and Hits@K. Ranking
requires all replacement head/tail candidates, not just the observed true
triples. Score these candidates through the frozen model cache and its shared
fusion policy, avoiding millions of redundant Mesa agent cycles. Reports
explicitly distinguish observed-triple agent execution from cached ranking.
Use --scores-only if only observed-triple agent messages are needed.

Load existing checkpoint and GA weights; no training or evolution takes place.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from vlkge import utils
from vlkge.ga.evaluation import build_score_cache
from vlkge.ga.run_fusion_ga import load_experiment, triples_to_ids
from vlkge.mas.run_ga_mas import create_ga_environment, load_fusion_weights


def parse_args(argv=None):
    """Select a full split, saved scorer/weights, and output destination.

    --summary prints a previously saved summary in ANN-style format without
    loading a checkpoint or rerunning agents. Otherwise the three scorer
    inputs are required. Validation is the default. --max-triples is a smoke
    limit; omit it for all rows. Candidate chunking bounds scoring intermediates
    while full ranking caches still require CPU memory for every query/entity.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", help="Print an existing summary.json without repeating evaluation")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fusion-weights")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=1024)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.summary is None:
        missing = [flag for flag, value in (
            ("--config", args.config), ("--checkpoint", args.checkpoint),
            ("--fusion-weights", args.fusion_weights),
        ) if not value]
        if missing:
            parser.error("Required unless --summary is supplied: " + ", ".join(missing))
    if args.candidate_batch_size < 1 or (args.max_triples is not None and args.max_triples < 1):
        parser.error("Candidate batch size and optional triple limit must be positive.")
    if args.decision_threshold is not None and not math.isfinite(args.decision_threshold):
        parser.error("--decision-threshold must be finite.")
    return args


def format_results(summary):
    """Format saved GA metrics like the existing ANN validation report.

    Print overall MRR/Hits@K with four decimals and named per-relation metrics
    with three decimals, using the same separators and relation-name padding
    as evaluate_mas.py. Values are only formatted, never recalculated. Derive
    Hits@K columns from the saved metrics so old reports need no config file.
    """
    if "metrics" not in summary:
        raise ValueError("This summary has no ranking metrics; evaluate without --scores-only first.")
    metrics = summary["metrics"]
    ks = sorted(int(key[5:]) for key in metrics if key.startswith("Hits@"))
    split = "Validation" if summary.get("split", "val") == "val" else "Test"
    relation_names = summary.get("relation_id_to_name", {})
    lines = [repr(summary.get("model", "MASGATransE")), "", "=" * 80,
             f"FINAL {split.upper()} RESULTS:", f"  MRR: {metrics['MRR']:.4f}"]
    lines.extend(f"  Hits@{k}: {metrics[f'Hits@{k}']:.4f}" for k in ks)
    lines.extend(["=" * 80, "", f"Per-Relation {split} Metrics:", "-" * 80])
    for relation_id, values in metrics.get("per_relation", {}).items():
        name = relation_names.get(str(relation_id), f"Relation_{relation_id}")
        hits = " | ".join(f"Hits@{k}: {values[f'Hits@{k}']:.3f}" for k in ks)
        lines.append(f"{name:30}: MRR: {values['MRR']:.3f}" + (f" | {hits}" if hits else ""))
    lines.extend(["-" * 80, "=" * 80])
    return "\n".join(lines)


def run_agent_rows(environment, model, loader, frame, output):
    """Execute the real agents for each observed triple and stream their output.

    publish_candidate clears the prior message queue. Write three messages
    per row, preserving duplicate triples and source row indexes. Comparing
    each combined score to direct forward detects inconsistent aggregation.
    This function reports scores/decisions, not link-prediction ranks.
    """
    fields = ["row_number", "source_index", "head", "relation", "tail", "visual_score",
              "textual_score", "combined_score", "decision", "decision_threshold"]
    counts = Counter()
    max_difference = 0.0
    with (output / "agent_scores.csv").open("w", encoding="utf-8", newline="") as csv_stream, \
            (output / "agent_messages.jsonl").open("w", encoding="utf-8") as messages_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(frame.itertuples(), start=1):
            candidate = {"head": row.head, "relation": row.relation, "tail": row.tail}
            environment.publish_candidate(candidate)
            environment.step()
            decision = environment.decisor_agent.last_message
            device = model.alpha_tensor.device
            with torch.no_grad():
                direct = model(
                    torch.tensor([loader.entity_to_id[row.head]], device=device),
                    torch.tensor([loader.relation_to_id[row.relation]], device=device),
                    torch.tensor([loader.entity_to_id[row.tail]], device=device),
                ).item()
            difference = abs(direct - decision["combined_score"])
            max_difference = max(max_difference, difference)
            if not math.isclose(direct, decision["combined_score"], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Agent/model fusion differs at source row {row.Index}.")
            counts[decision["decision"]] += 1
            writer.writerow({
                "row_number": number, "source_index": int(row.Index), **candidate,
                **{key: decision[key] for key in fields[5:]},
            })
            messages_stream.write(json.dumps({
                "row_number": number, "source_index": int(row.Index),
                "messages": environment.messages,
            }, allow_nan=False) + "\n")
            if number % 100 == 0 or number == len(frame):
                print(f"Agents: {number}/{len(frame)} triples", flush=True)
    return {"agent_cycles": len(frame), "decision_counts": dict(counts),
            "max_agent_model_score_difference": max_difference}


def main(argv=None):
    """Load once, run all observed triples, then optionally compute ranks.

    Save summary.json with selected/effective weights, filtering rules, split
    size, agent counts, parity check, and optional ranking metrics. Equal-weight
    metrics use the same cache for comparison with the alpha=0.5 checkpoint.
    Test targets are accessed only when --split test is explicitly requested.
    """
    args = parse_args(argv)
    if args.summary:
        with open(args.summary, encoding="utf-8") as stream:
            summary = json.load(stream)
        print(format_results(summary))
        return summary
    utils.set_seed(args.seed)
    device = torch.device("cpu") if args.no_cuda else utils.set_gpu()
    weights = load_fusion_weights(args.fusion_weights, args.checkpoint)
    model, loader, config = load_experiment(args.config, args.checkpoint, device)
    model.set_alpha(weights["visual_weight"])
    environment = create_ga_environment(
        model, loader.entity_to_id, loader.relation_to_id, device,
        seed=args.seed, decision_threshold=args.decision_threshold,
    )
    _, validation, test = loader.split_data()
    frame = validation if args.split == "val" else test
    if args.max_triples is not None:
        frame = frame.iloc[:args.max_triples]
    if frame.empty:
        raise ValueError(f"The {args.split} split has no evaluation triples.")
    output = (Path(args.output_dir) if args.output_dir else
              Path(args.fusion_weights).resolve().parent / f"agents_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    agent_results = run_agent_rows(environment, model, loader, frame, output)
    alpha = float(model.alpha_tensor.item())
    summary = {
        "model": "MASGATransE", "split": args.split, "triples": len(frame),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": weights["checkpoint_sha256"],
        "fusion_weights_file": str(Path(args.fusion_weights).resolve()),
        "visual_weight": alpha, "textual_weight": 1 - alpha,
        "decision_threshold": args.decision_threshold, "run_arguments": vars(args),
        "agent_execution": "three actual agents per observed triple",
        **agent_results,
    }
    if not args.scores_only:
        print("Calculating all-candidate ranking scores with the shared frozen policy...", flush=True)
        cache = build_score_cache(
            model, triples_to_ids(frame, loader),
            loader.data[["hid", "rid", "tid"]].to_numpy(dtype=np.int64),
            bidirectional=config["bidirectional_eval"],
            candidate_batch_size=args.candidate_batch_size,
            progress=lambda done, total: print(f"Ranking: {done}/{total} queries", flush=True),
        )
        summary.update({
            "ranking_execution": "cached model scores with shared agent/model fusion policy",
            "filtering": "direction-specific all-known filters; all entities are candidates",
            "tie_policy": "score descending, entity ID ascending",
            "bidirectional": config["bidirectional_eval"],
            "metrics": cache.metrics(alpha, ks=config["top_k"]),
            "equal_weight_metrics": cache.metrics(0.5, ks=config["top_k"]),
            "relation_id_to_name": {str(value): key for key, value in loader.relation_to_id.items()},
        })
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"Agent cycles: {len(frame)}; visual={alpha:.6f}, textual={1-alpha:.6f}")
    print(f"Results: {output.resolve()}")
    if "metrics" in summary:
        report = format_results(summary)
        (output / "results.txt").write_text(report + "\n", encoding="utf-8")
        print(report)
    return summary


if __name__ == "__main__":
    main()
