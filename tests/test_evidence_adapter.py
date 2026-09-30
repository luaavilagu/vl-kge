import unittest

import torch

from vlkge.mas.evidence_adapter import EvidenceAdapter


class FakeEvidenceModel:
    """Small deterministic model used to test the adapter only."""

    def __init__(self):
        self.in_evaluation_mode = False

    def eval(self):
        self.in_evaluation_mode = True
        return self

    def score_visual_evidence(self, head, relation, tail):
        return (head + 2 * relation + 3 * tail).float()

    def score_textual_evidence(self, head, relation, tail):
        return (3 * head + relation + tail).float()


class EvidenceAdapterTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeEvidenceModel()
        self.adapter = EvidenceAdapter(
            model=self.model,
            entity_to_id={"head_entity": 2, "tail_entity": 7},
            relation_to_id={"related_to": 5},
            device=torch.device("cpu"),
        )
        self.candidate = {
            "head": "head_entity",
            "relation": "related_to",
            "tail": "tail_entity",
        }

    def test_candidate_to_tensors(self):
        head, relation, tail = self.adapter.candidate_to_tensors(self.candidate)

        self.assertEqual(head.tolist(), [2])
        self.assertEqual(relation.tolist(), [5])
        self.assertEqual(tail.tolist(), [7])
        self.assertEqual(head.dtype, torch.long)
        self.assertEqual(head.device.type, "cpu")

    def test_returns_visual_and_textual_scores(self):
        self.assertEqual(self.adapter.score_visual(self.candidate), 33.0)
        self.assertEqual(self.adapter.score_textual(self.candidate), 18.0)
        self.assertTrue(self.model.in_evaluation_mode)

    def test_rejects_unknown_entities(self):
        invalid_candidate = {**self.candidate, "head": "unknown_entity"}

        with self.assertRaisesRegex(ValueError, "Unknown head entity"):
            self.adapter.candidate_to_tensors(invalid_candidate)


if __name__ == "__main__":
    unittest.main()
