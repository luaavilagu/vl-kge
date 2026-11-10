"""Run visual, textual, and decision agents with a frozen MASGATransE scorer.

Supply the training configuration, original scoring checkpoint, and the
GA's fusion_weights.json (fusion_result.json is also accepted). Both saved
formats identify the checkpoint by SHA-256. The selected alpha is restored
before constructing agents; they never run evolution or relation training.

Example, from the repository root:
    python -m vlkge.mas.run_ga_mas --config CONFIG --checkpoint CHECKPOINT
        --fusion-weights WEIGHTS --head ENTITY --relation RELATION --tail ENTITY

The default reports raw evidence and the weighted score as unclassified.
--decision-threshold enables binary decisions using a caller-supplied cutoff.
GA validation MRR does not itself establish a suitable cutoff.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from vlkge import utils
from vlkge.ga.run_fusion_ga import file_sha256, load_experiment
from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.evidence_adapter import EvidenceAdapter


def load_fusion_weights(weights_path, checkpoint_path):
    """Read bounded complementary weights and verify checkpoint provenance.

    GA search exports both a small weights file and a full report. Both have
    the required visual_weight, textual_weight, and checkpoint_sha256 fields.
    Reject invalid weights instead of silently clipping them or selecting
    equal weights, because either would change the searched scoring policy.
    """
    with open(weights_path, encoding="utf-8") as stream:
        weights = json.load(stream)
    if not isinstance(weights, dict):
        raise ValueError("The fusion weights file must contain a JSON object.")
    try:
        visual = float(weights["visual_weight"])
        textual = float(weights["textual_weight"])
        checkpoint_hash = weights["checkpoint_sha256"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Fusion weights require visual_weight, textual_weight, and checkpoint_sha256.") from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (visual, textual)):
        raise ValueError("Both fusion weights must be finite and in [0, 1].")
    if not math.isclose(visual + textual, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Visual and textual weights must sum to one.")
    if checkpoint_hash != file_sha256(checkpoint_path):
        raise ValueError("Fusion weights belong to a different scoring checkpoint.")
    return {"visual_weight": visual, "textual_weight": textual,
            "checkpoint_sha256": checkpoint_hash}


def create_ga_environment(model, entity_to_id, relation_to_id, device,
                          seed=42, decision_threshold=None):
    """Connect an already loaded scorer to the three agents.

    Reuse this for single-candidate inference and full-split evaluation so
    both entry points execute the same evidence and decision-agent policies.
    The caller is responsible for loading alpha and freezing model weights.
    """
    adapter = EvidenceAdapter(
        model=model, entity_to_id=entity_to_id,
        relation_to_id=relation_to_id, device=device,
    )
    return KnowledgeGraphEnvironment(
        visual_evidence_model=adapter.score_visual,
        textual_evidence_model=adapter.score_textual,
        score_fusion=adapter.fuse_scores,
        decision_threshold=decision_threshold,
        seed=seed,
    )


def build_ga_environment(config_path, checkpoint_path, weights_path, device,
                         seed=42, decision_threshold=None):
    """Load frozen evidence and expose model fusion through the agent adapter.

    The visual and textual agents still emit independent scores. Only the
    decision agent calls adapter.fuse_scores, which delegates to the model's
    registered alpha buffer. This keeps evidence production separate from
    aggregation while ensuring both direct inference and agents agree.

    Return the environment, scoring model, and verified saved weights so
    callers can inspect provenance and compare agent/model outputs.
    """
    weights = load_fusion_weights(weights_path, checkpoint_path)
    model, loader, _ = load_experiment(config_path, checkpoint_path, device)
    model.set_alpha(weights["visual_weight"])
    environment = create_ga_environment(
        model, loader.entity_to_id, loader.relation_to_id, device,
        seed=seed, decision_threshold=decision_threshold,
    )
    return environment, model, weights


def parse_args(argv=None):
    """Accept the trained scorer, GA weights, and one named candidate triple."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fusion-weights", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.decision_threshold is not None and not math.isfinite(args.decision_threshold):
        parser.error("--decision-threshold must be finite.")
    return args


def main(argv=None):
    """Execute one agent cycle and report scores, effective weights, and policy.

    No files or model parameters are changed. An unclassified decision means
    the numeric fused score is available, but no binary cutoff was supplied.
    Alpha is reported from the float32 model buffer used in the calculation.
    """
    args = parse_args(argv)
    utils.set_seed(args.seed)
    device = torch.device("cpu") if args.no_cuda else utils.set_gpu()
    environment, model, weights = build_ga_environment(
        args.config, args.checkpoint, args.fusion_weights, device,
        seed=args.seed, decision_threshold=args.decision_threshold,
    )
    candidate = {"head": args.head, "relation": args.relation, "tail": args.tail}
    environment.publish_candidate(candidate)
    environment.step()
    visual_weight = float(model.alpha_tensor.item())
    result = {
        "model": "MASGATransE", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": weights["checkpoint_sha256"],
        "visual_weight": visual_weight, "textual_weight": 1 - visual_weight,
        "decision_threshold": args.decision_threshold,
        "messages": environment.messages,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
