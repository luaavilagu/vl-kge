"""Check GA-weighted agent inference, legacy behavior, and saved-weight validation."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from vlkge.ga.run_fusion_ga import file_sha256
from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.evidence_adapter import EvidenceAdapter
from vlkge.mas.run_ga_mas import load_fusion_weights
from vlkge.models.mas_gatranse import MASGATransE


class GAAgentTests(unittest.TestCase):
    """Verify the decision agent uses the model's actual alpha policy."""

    def make_environment(self, threshold=None):
        """Create a frozen scorer and real Mesa agents using tiny features."""
        visual = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        textual = torch.tensor([[0., 1.], [1., 0.], [0., -1.]])
        model = MASGATransE(
            3, 1, 2, visual_features=visual, textual_features=textual,
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
        )
        with torch.no_grad():
            model.relation_embeddings.weight.copy_(torch.tensor([[0.8, 0.4]]))
        model.requires_grad_(False)
        adapter = EvidenceAdapter(model, {"a": 0, "b": 1, "c": 2}, {"r": 0}, torch.device("cpu"))
        environment = KnowledgeGraphEnvironment(
            adapter.score_visual, adapter.score_textual,
            score_fusion=adapter.fuse_scores, decision_threshold=threshold,
        )
        return environment, model

    def test_agents_match_forward_and_keep_parameters_frozen(self):
        environment, model = self.make_environment()
        relations_before = model.relation_embeddings.weight.detach().clone()
        for alpha in (0.0, 0.37, 0.5, 1.0):
            model.set_alpha(alpha)
            environment.publish_candidate({"head": "a", "relation": "r", "tail": "b"})
            environment.step()
            decision = environment.decisor_agent.last_message
            expected = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1])).item()
            self.assertEqual(decision["combined_score"], expected)
            self.assertEqual(decision["decision"], "unclassified")
            self.assertIsNone(decision["decision_threshold"])
            self.assertEqual([m["type"] for m in environment.messages],
                             ["visual_evidence", "textual_evidence", "link_decision"])
        torch.testing.assert_close(model.relation_embeddings.weight, relations_before)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_optional_threshold_and_original_environment_defaults(self):
        environment = KnowledgeGraphEnvironment(lambda candidate: 8., lambda candidate: 4.)
        candidate = {"head": "a", "relation": "r", "tail": "b"}
        environment.publish_candidate(candidate)
        environment.step()
        self.assertEqual(environment.decisor_agent.combined_score, 6.)
        self.assertEqual(environment.decisor_agent.decision_message, "accepted")
        environment, _ = self.make_environment(threshold=100.)
        environment.publish_candidate(candidate)
        environment.step()
        self.assertEqual(environment.decisor_agent.decision_message, "rejected")

    def test_missing_new_evidence_does_not_reuse_previous_candidate(self):
        environment, _ = self.make_environment()
        environment.publish_candidate({"head": "a", "relation": "r", "tail": "b"})
        environment.step()
        environment.publish_candidate({"head": "b", "relation": "r", "tail": "c"})
        environment.visual_agent.step()
        with self.assertRaisesRegex(ValueError, "Both visual and textual"):
            environment.decisor_agent.step()

    def test_saved_weights_check_bounds_sum_and_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="ga_weights_") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
            path = root / "weights.json"
            weights = {"visual_weight": 0.7, "textual_weight": 0.3,
                       "checkpoint_sha256": file_sha256(checkpoint)}
            path.write_text(json.dumps(weights), encoding="utf-8")
            self.assertEqual(load_fusion_weights(path, checkpoint)["visual_weight"], 0.7)
            for changed in (
                {"visual_weight": -0.1}, {"textual_weight": 0.8},
                {"checkpoint_sha256": "wrong checkpoint"},
            ):
                path.write_text(json.dumps({**weights, **changed}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_fusion_weights(path, checkpoint)

    def test_shared_fusion_preserves_relation_gradients_for_training(self):
        """Agent reuse must not detach the scoring path used before freezing."""
        _, model = self.make_environment()
        model.relation_embeddings.weight.requires_grad_(True)
        model.set_alpha(0.37)
        model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1])).sum().backward()
        gradient = model.relation_embeddings.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(gradient.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
