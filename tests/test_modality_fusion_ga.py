"""Focused checks for operators, ranking, and the train/checkpoint/search path.

Use ``python -m unittest discover -s tests -p test_modality_fusion_ga.py``.
The integration case uses a tiny temporary dataset; it does not train WN9-IMG.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from vlkge import helpers
from vlkge.dataloader import KGDataset
from vlkge.ga.evaluation import build_score_cache, directional_filters
from vlkge.ga.modality_fusion_ga import GAConfig, arithmetic_crossover, run_ga
from vlkge.ga.run_fusion_ga import file_sha256, load_experiment, main as run_search
from vlkge.models.mas_gatranse import MASGATransE
from vlkge.mas.run_ga_mas import build_ga_environment
from vlkge.scripts.train import main as train_model


class EvolutionTests(unittest.TestCase):
    """Exercise bounds, elitism, stopping, and seeded reproducibility."""

    def test_seeded_search_preserves_elites_and_random_state(self):
        config = GAConfig(population_size=8, generations=6, patience=None)
        state = random.getstate()
        first = run_ga(lambda a: 1 - (a - 0.75) ** 2, config)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, run_ga(lambda a: 1 - (a - 0.75) ** 2, config))
        self.assertGreaterEqual(first["best_mrr"], 0.9375)
        history = [row["best_mrr"] for row in first["history"]]
        self.assertEqual(history, sorted(history))
        self.assertGreaterEqual(first["fitness_requests"], first["unique_evaluations"])

    def test_invalid_ga_configuration_is_rejected(self):
        """Validation concerns legal settings, not whether defaults are optimal."""
        for options in (
            {"population_size": 2}, {"tournament_size": 0},
            {"elite_size": 30}, {"mutation_probability": float("nan")},
            {"crossover_probability": 1.1}, {"patience": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GAConfig(**options)

    def test_arithmetic_uses_original_parents_and_stays_bounded(self):
        first, second = [0.2], [0.8]
        with patch("vlkge.ga.modality_fusion_ga.random.random", return_value=0.25):
            arithmetic_crossover(first, second)
        self.assertAlmostEqual(first[0], 0.65)
        self.assertAlmostEqual(second[0], 0.35)

    def test_sbx_reset_bounds_and_flat_fitness_stopping(self):
        seen = []

        def fitness(alpha):
            self.assertTrue(0 <= alpha <= 1)
            seen.append(alpha)
            return 0.5

        result = run_ga(fitness, GAConfig(
            population_size=8, generations=10, crossover="sbx",
            crossover_probability=1, mutation_probability=1, patience=2,
        ))
        self.assertEqual(result["generations_completed"], 2)
        self.assertEqual(result["stopping_reason"], "no_improvement")
        self.assertTrue({0.0, 0.5, 1.0}.issubset(seen))


class RankingTests(unittest.TestCase):
    """Compare cached ranks to actual model inference in both directions."""

    def test_cache_matches_direct_filtered_ranking_at_endpoints_and_interior(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        model = MASGATransE(
            3, 1, 2, visual_features=features, textual_features=features.flip(0),
            visual_entity_to_index={0: 0, 1: 1, 2: 2},
            textual_entity_to_index={0: 0, 1: 1, 2: 2},
        )
        with torch.no_grad():
            model.relation_embeddings.weight.copy_(torch.tensor([[0.8, 0.4]]))
        triples = np.array([[0, 0, 1], [2, 0, 0]], dtype=np.int64)
        known = np.array([[0, 0, 1], [2, 0, 0], [0, 0, 2]], dtype=np.int64)
        tails, heads = directional_filters(known)
        self.assertEqual(tails[(0, 0)], {1, 2})
        self.assertEqual(heads[(0, 0)], {2})
        cached = build_score_cache(model, triples, known, candidate_batch_size=2)
        frame = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        loader = DataLoader(KGDataset(frame, {0: 0, 1: 1, 2: 2}, {0: 0}), batch_size=2)
        for alpha in (0.0, 0.37, 0.5, 1.0):
            model.set_alpha(alpha)
            expected_mrr, expected_hits, _ = helpers.evaluate_kge(
                model, loader, tails, head_filter_map=heads
            )
            actual = cached.metrics(alpha)
            self.assertAlmostEqual(actual["MRR"], expected_mrr)
            for k, expected in expected_hits.items():
                self.assertAlmostEqual(actual[f"Hits@{k}"], expected)


class WorkflowTests(unittest.TestCase):
    """Train a tiny scorer, search its frozen checkpoint, and read reports."""

    def test_training_checkpoint_search_and_saved_weights(self):
        with tempfile.TemporaryDirectory(prefix="fusion_ga_") as directory:
            root = Path(directory)
            entities = ["a", "b", "c", "d"]
            rows = [(entities[i], "r", entities[(i + 1) % 4], "train")
                    for _ in range(3) for i in range(4)]
            rows += [("a", "r", "c", "val"), ("d", "r", "b", "val"),
                     ("b", "r", "d", "test"), ("c", "r", "a", "test")]
            data_path = root / "triples.csv"
            pd.DataFrame(rows, columns=["head", "relation", "tail", "mode"]).to_csv(data_path, index=False)
            vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
            for name, values in (("visual", vectors), ("textual", vectors[::-1])):
                with (root / f"{name}.pkl").open("wb") as stream:
                    pickle.dump(dict(zip(entities, values)), stream)
            checkpoint_path = root / "scorer.pt"
            config = {
                "dataset": "wn9_img", "data_path": str(data_path), "model": "MASGATransE",
                "visual_features_path": str(root / "visual.pkl"),
                "textual_features_path": str(root / "textual.pkl"),
                "embedding_dim": 2, "fusion_mode": "average", "use_structural": False,
                "use_visual": True, "use_textual": True, "use_relation_features": False,
                "freeze_visual": True, "freeze_textual": True,
                "visual_proj": False, "textual_proj": False,
                "normalize_visual": False, "normalize_textual": False,
                "epochs": 1, "batch_size": 4, "num_neg_samples": 1,
                "bidirectional_eval": True, "skip_final_test": True,
                "save_path": str(checkpoint_path), "seed": 42,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["train", "--config", str(config_path), "--no_cuda"]):
                    train_model()
                model, _, _ = load_experiment(config_path, checkpoint_path, torch.device("cpu"))
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                original_hash = file_sha256(checkpoint_path)
                arguments = [
                    "--config", str(config_path), "--checkpoint", str(checkpoint_path),
                    "--population-size", "6", "--generations", "2", "--patience", "0",
                    "--candidate-batch-size", "2", "--no-cuda", "--evaluate-test",
                    "--output-dir", str(root / "search"),
                ]
                run_search(arguments)
                report_path = root / "search" / "fusion_result.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["validation_triples"], 2)
                self.assertEqual(report["test_triples"], 2)
                self.assertEqual(len(report["ga"]["history"]), 3)
                self.assertIn("test", report["methods"]["ga"])
                self.assertEqual(report["methods"]["random_search"]["evaluations"],
                                 report["ga"]["unique_evaluations"])
                self.assertAlmostEqual(report["visual_weight"] + report["textual_weight"], 1)
                self.assertEqual(file_sha256(checkpoint_path), original_hash)
                environment, agent_model, _ = build_ga_environment(
                    config_path, checkpoint_path, root / "search" / "fusion_weights.json",
                    torch.device("cpu"),
                )
                environment.publish_candidate({"head": "a", "relation": "r", "tail": "c"})
                environment.step()
                adapter = environment.visual_evidence_model.__self__
                head, relation, tail = adapter.candidate_to_tensors(environment.current_candidate)
                self.assertEqual(environment.decisor_agent.combined_score,
                                 agent_model(head, relation, tail).item())
                self.assertEqual(environment.decisor_agent.decision_message, "unclassified")
                run_search(arguments + ["--weights", str(report_path), "--output-dir", str(root / "reuse")])
                reused = json.loads((root / "reuse" / "fusion_result.json").read_text(encoding="utf-8"))
                self.assertEqual(report["visual_weight"], reused["visual_weight"])


if __name__ == "__main__":
    unittest.main()
