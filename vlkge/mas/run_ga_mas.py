"""Run visual, textual, structural, and decision agents with frozen GA fusion.

Supply the training configuration, original scoring checkpoint, and the
GA's fusion_weights.json (fusion_result.json is also accepted). Both saved
formats identify the checkpoint by SHA-256. Three selected weights are restored
before constructing agents; they never run evolution or relation training.

Example, from the repository root:
    python -m vlkge.mas.run_ga_mas --config CONFIG --checkpoint CHECKPOINT
        --fusion-weights WEIGHTS --head ENTITY --relation RELATION --tail ENTITY

The optional decision test defaults to the requested threshold 0.5. It has no
effect on the MRR objective or candidate ranking.
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
from vlkge.models.mas_gatranse import MODALITIES, validate_weights


def load_fusion_weights(weights_path, checkpoint_path):
    """Read three named simplex weights and verify checkpoint provenance.

    GA search exports both a small weights file and a full report. Both have
    format_version=2, modality_order, weights, and checkpoint_sha256 fields.
    Reject invalid weights instead of silently clipping them or selecting
        equal weights, because either would change the searched scoring policy.
    """
    with open(weights_path, encoding="utf-8") as stream:
        weights = json.load(stream)
    if not isinstance(weights, dict):
        raise ValueError("The fusion weights file must contain a JSON object.")
    if weights.get("format_version") != 2 or weights.get("modality_order") != list(MODALITIES):
        raise ValueError("Expected three-modality weights with format_version=2 and explicit modality_order.")
    try:
        values = validate_weights(weights["weights"])
        checkpoint_hash = weights["checkpoint_sha256"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Expected valid visual/textual/structural weights and checkpoint_sha256.") from error
    for name, value in zip(MODALITIES, values):
        if f"{name}_weight" in weights and not math.isclose(
                float(weights[f"{name}_weight"]), value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Conflicting {name} weights in the saved file.")
    if checkpoint_hash != file_sha256(checkpoint_path):
        raise ValueError("Fusion weights belong to a different scoring checkpoint.")
    return {"format_version": 2, "modality_order": list(MODALITIES),
            "weights": dict(zip(MODALITIES, values)),
            **{f"{name}_weight": value for name, value in zip(MODALITIES, values)},
            "checkpoint_sha256": checkpoint_hash}


def create_ga_environment(model, entity_to_id, relation_to_id, device,
                          seed=42, decision_threshold=0.5):
    """Connect an already loaded scorer to three evidence agents and a decisor.

    Reuse this for single-candidate inference and full-split evaluation so
    both entry points execute the same evidence and decision-agent policies.
    The caller loads all three weights and freezes every learned parameter.
    """
    adapter = EvidenceAdapter(
        model=model, entity_to_id=entity_to_id,
        relation_to_id=relation_to_id, device=device,
    )
    return KnowledgeGraphEnvironment(
        visual_evidence_model=adapter.score_visual,
        textual_evidence_model=adapter.score_textual,
        structural_evidence_model=adapter.score_structural,
        score_fusion=adapter.fuse_scores,
        decision_threshold=decision_threshold,
        seed=seed,
    )


def build_ga_environment(config_path, checkpoint_path, weights_path, device,
                         seed=42, decision_threshold=0.5):
    """Load frozen evidence and expose model fusion through the agent adapter.

    The visual, textual, and structural agents emit separate scores. Only the
    decision agent calls adapter.fuse_scores, which delegates to the model's
    registered three-weight buffer. This keeps evidence production separate from
    aggregation while ensuring both direct inference and agents agree.

    Return the environment, scoring model, and verified saved weights so
    callers can inspect provenance and compare agent/model outputs.
    """
    weights = load_fusion_weights(weights_path, checkpoint_path)
    model, loader, _ = load_experiment(config_path, checkpoint_path, device)
    model.set_weights(weights["weights"])
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
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.decision_threshold is not None and not math.isfinite(args.decision_threshold):
        parser.error("--decision-threshold must be finite.")
    return args


def main(argv=None):
    """Execute one agent cycle and report scores, effective weights, and policy.

    No files or model parameters are changed. An unclassified decision means
    the numeric fused score is available, but no binary cutoff was supplied.
    All weights are reported from the model buffer used in the calculation.
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
    effective = dict(zip(MODALITIES, model.modality_weights.detach().cpu().tolist()))
    result = {
        "model": "MASGATransE", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": weights["checkpoint_sha256"],
        "format_version": 2, "modality_order": list(MODALITIES), "weights": effective,
        **{f"{name}_weight": value for name, value in effective.items()},
        "decision_threshold": args.decision_threshold,
        "messages": environment.messages,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
