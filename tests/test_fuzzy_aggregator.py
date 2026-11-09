"""Tests for fuzzy evidence aggregation and agent integration."""

import unittest

from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.fuzzy_aggregator import FuzzyEvidenceAggregator


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


if __name__ == "__main__":
    unittest.main()
