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

    Validation is the default. --max-triples is an explicitly recorded smoke
    limit; omit it for all rows. Candidate chunking bounds scoring intermediates
    while full ranking caches still require CPU memory for every query/entity.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fusion-weights", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=1024)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.candidate_batch_size < 1 or (args.max_triples is not None and args.max_triples < 1):
        parser.error("Candidate batch size and optional triple limit must be positive.")
    if args.decision_threshold is not None and not math.isfinite(args.decision_threshold):
        parser.error("--decision-threshold must be finite.")
    return args


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
        metrics = summary["metrics"]
        print(f"{args.split.upper()} MRR: {metrics['MRR']:.6f}")
        for k in config["top_k"]:
            print(f"Hits@{k}: {metrics[f'Hits@{k}']:.6f}")
        print(f"Equal-weight MRR: {summary['equal_weight_metrics']['MRR']:.6f}")
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"Agent cycles: {len(frame)}; visual={alpha:.6f}, textual={1-alpha:.6f}")
    print(f"Results: {output.resolve()}")
    return summary


if __name__ == "__main__":
    main()
