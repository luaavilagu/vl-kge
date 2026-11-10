"""Tests for fuzzy evidence aggregation and agent integration."""

import unittest

import torch

from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.evidence_adapter import EvidenceAdapter
from vlkge.mas.fuzzy_aggregator import FuzzyEvidenceAggregator
from vlkge import helpers
from vlkge.models import TransEFuzzy as ExportedTransEFuzzy
from vlkge.models.transe import TransE
from vlkge.models.transe_fuzzy import TransEFuzzy


class FuzzyAggregatorTests(unittest.TestCase):
    def test_consensus_scores_are_bounded_and_linguistically_labeled(self):
        aggregator = FuzzyEvidenceAggregator()
        low = aggregator.evaluate(0.0, 0.0, 0.0)
        medium = aggregator.evaluate(0.5, 0.5, 0.5)
        high = aggregator.evaluate(1.0, 1.0, 1.0)

        self.assertEqual(low.label, "low")
        self.assertEqual(medium.label, "medium")
        self.assertEqual(high.label, "high")
        self.assertLess(low.score, medium.score)
        self.assertLess(medium.score, high.score)
        for result in (low, medium, high):
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_calibration_ranges_normalize_raw_scores_and_reject_outliers(self):
        aggregator = FuzzyEvidenceAggregator({
            "visual": (-2.0, 2.0),
            "textual": (10.0, 20.0),
            "structural": (-5.0, 5.0),
        })
        result = aggregator.evaluate(2.0, 20.0, 5.0)
        self.assertEqual(result.label, "high")
        with self.assertRaisesRegex(ValueError, "outside calibrated range"):
            aggregator.evaluate(2.1, 20.0, 5.0)

    def test_aggregator_is_connected_to_decisor_agent(self):
        candidate = {"head": "h", "relation": "r", "tail": "t"}
        aggregator = FuzzyEvidenceAggregator()
        environment = KnowledgeGraphEnvironment(
            visual_evidence_model=lambda _: 0.9,
            textual_evidence_model=lambda _: 0.8,
            structural_evidence_model=lambda _: 1.0,
            decision_threshold=None,
            fuzzy_aggregator=aggregator,
        )
        environment.publish_candidate(candidate)
        environment.step()

        decision = environment.decisor_agent.last_message
        expected = aggregator.evaluate(0.9, 0.8, 1.0)
        self.assertEqual(decision["combined_score"], expected.score)
        self.assertEqual(decision["fuzzy_confidence"], expected.label)
        self.assertEqual(decision["decision"], "unclassified")

    def test_fuzzy_and_legacy_fusion_cannot_both_be_selected(self):
        with self.assertRaisesRegex(ValueError, "either score_fusion or fuzzy_aggregator"):
            KnowledgeGraphEnvironment(
                lambda _: 0.5, lambda _: 0.5, lambda _: 0.5,
                score_fusion=lambda visual, textual, structural: visual,
                fuzzy_aggregator=FuzzyEvidenceAggregator(),
            )

    def test_invalid_scores_and_ranges_are_reported(self):
        aggregator = FuzzyEvidenceAggregator()
        with self.assertRaisesRegex(ValueError, "must be finite"):
            aggregator.evaluate(float("nan"), 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "finite and increasing"):
            FuzzyEvidenceAggregator({"visual": (1.0, 1.0)})

    def test_adapter_derives_three_scores_from_base_transe_checkpoint(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        model = TransE(
            3, 1, 2,
            visual_features=features,
            textual_features=features.flip(0),
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
            use_structural=True,
            use_visual=True,
            use_textual=True,
        )
        with torch.no_grad():
            model.entity_embeddings.weight.copy_(torch.tensor([
                [1., 0.], [0., -1.], [-1., 0.]
            ]))
            model.relation_embeddings.weight.copy_(torch.tensor([[0.2, 0.4]]))
        adapter = EvidenceAdapter(
            model, {"a": 0, "b": 1, "c": 2}, {"r": 0}, torch.device("cpu")
        )
        candidate = {"head": "a", "relation": "r", "tail": "b"}

        with torch.no_grad():
            original_score = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        scores = (
            adapter.score_visual(candidate),
            adapter.score_textual(candidate),
            adapter.score_structural(candidate),
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(score)) for score in scores))
        with torch.no_grad():
            unchanged_score = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        torch.testing.assert_close(original_score, unchanged_score)
        self.assertEqual(len(set(round(score, 6) for score in scores)), 3)

    def test_base_transe_scores_flow_through_agents_into_fuzzy_decision(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        model = TransE(
            3, 1, 2,
            visual_features=features,
            textual_features=features.flip(0),
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
            use_structural=True,
            use_visual=True,
            use_textual=True,
        )
        adapter = EvidenceAdapter(
            model, {"a": 0, "b": 1, "c": 2}, {"r": 0}, torch.device("cpu")
        )
        aggregator = FuzzyEvidenceAggregator({
            "visual": (-10.0, 12.0),
            "textual": (-10.0, 12.0),
            "structural": (-10.0, 12.0),
        })
        environment = KnowledgeGraphEnvironment(
            visual_evidence_model=adapter.score_visual,
            textual_evidence_model=adapter.score_textual,
            structural_evidence_model=adapter.score_structural,
            fuzzy_aggregator=aggregator,
            decision_threshold=None,
        )
        candidate = {"head": "a", "relation": "r", "tail": "b"}
        environment.publish_candidate(candidate)
        environment.step()

        messages = {message["type"]: message for message in environment.messages}
        decision = environment.decisor_agent.last_message
        expected = aggregator.evaluate(
            messages["visual_evidence"]["score"],
            messages["textual_evidence"]["score"],
            messages["structural_evidence"]["score"],
        )
        self.assertEqual(decision["combined_score"], expected.score)
        self.assertEqual(decision["fuzzy_confidence"], expected.label)

    def test_transe_fuzzy_score_is_in_training_graph(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        model = TransEFuzzy(
            3, 1, 2,
            visual_features=features,
            textual_features=features.flip(0),
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
            use_structural=True,
            use_visual=True,
            use_textual=True,
            freeze_visual=False,
            freeze_textual=False,
            p_norm=2,
            raw_margin=2.0,
        )
        positive = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        negative = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]))
        loss = torch.nn.functional.softplus(-positive).mean()
        loss = loss + torch.nn.functional.softplus(negative).mean()
        loss.backward()

        self.assertEqual(positive.shape, (1,))
        self.assertTrue(torch.isfinite(positive).all())
        self.assertTrue(torch.isfinite(negative).all())
        self.assertIsNotNone(model.fuzzy_distance_thresholds.grad)
        self.assertTrue(torch.isfinite(model.fuzzy_distance_thresholds.grad).all())
        self.assertIsNotNone(model.relation_embeddings.weight.grad)
        self.assertIsNotNone(model.entity_embeddings.weight.grad)
        self.assertIsNotNone(model.visual_embeddings.weight.grad)
        self.assertIsNotNone(model.textual_embeddings.weight.grad)
        self.assertGreater(model.fuzzy_distance_thresholds.grad.abs().sum().item(), 0.0)
        thresholds_before = model.fuzzy_distance_thresholds.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.step()
        self.assertFalse(torch.equal(
            thresholds_before, model.fuzzy_distance_thresholds.detach()
        ))

    def test_transe_fuzzy_requires_all_modalities(self):
        with self.assertRaisesRegex(ValueError, "requires structural, visual, and textual"):
            TransEFuzzy(3, 1, 2)

    def test_transe_fuzzy_rejects_concat_relation_dimension(self):
        features = torch.eye(3, 2)
        with self.assertRaisesRegex(ValueError, "does not support concat"):
            TransEFuzzy(
                3, 1, 2,
                visual_features=features,
                textual_features=features,
                visual_entity_to_index={0: 0, 1: 1, 2: 2},
                textual_entity_to_index={0: 0, 1: 1, 2: 2},
                use_structural=True,
                use_visual=True,
                use_textual=True,
                fusion_mode="concat",
            )

    def test_base_transe_remains_non_fuzzy(self):
        model = TransE(3, 1, 2, use_structural=True)
        self.assertFalse(hasattr(model, "fuzzy_distance_thresholds"))
        score = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(score.shape, (1,))

    def test_training_factory_exposes_separate_transefuzzy_model(self):
        features = torch.eye(3, 2)
        model, _, _ = helpers.get_model(
            model_name="TransEFuzzy",
            num_entities=3,
            num_relations=1,
            embedding_dim=2,
            visual_features=features,
            textual_features=features,
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
            use_structural=True,
            use_visual=True,
            use_textual=True,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(model, TransEFuzzy)
        self.assertIs(ExportedTransEFuzzy, TransEFuzzy)


if __name__ == "__main__":
    unittest.main()
