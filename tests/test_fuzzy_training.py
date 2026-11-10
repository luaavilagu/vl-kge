"""Regression checks for fuzzy training, memory use, and checkpoint integration."""
import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from vlkge import helpers
from vlkge.models import TransEFuzzy
from vlkge.mas.fuzzy_aggregator import FuzzyEvidenceAggregator
from vlkge.mas.run_fuzzy_transe import load_transe_fuzzy_environment
from vlkge.scripts.train import main, parse_args


class FuzzyTrainingTests(unittest.TestCase):
    def model(self, **kwargs):
        torch.manual_seed(42)
        features = torch.randn(4, 8)
        options = dict(visual_features=features, textual_features=features.flip(0),
                       visual_entity_to_index=dict(enumerate(range(4))),
                       textual_entity_to_index=dict(enumerate(range(4))),
                       use_visual=True, use_textual=True,
                       freeze_visual=False, freeze_textual=False)
        options.update(kwargs)
        return TransEFuzzy(4, 1, 8, **options)

    def test_plateau_still_has_gradients_and_distinct_scores(self):
        model = self.model()
        with torch.no_grad():
            model.fuzzy_distance_thresholds.fill_(-3)
        scores = model(torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1, 2]))
        torch.nn.functional.softplus(-scores).mean().backward()
        self.assertNotEqual(scores[0].item(), scores[1].item())
        for parameter in (model.fuzzy_distance_thresholds, model.entity_embeddings.weight,
                          model.relation_embeddings.weight, model.visual_embeddings.weight,
                          model.textual_embeddings.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_grouped_rules_preserve_mamdani_scores(self):
        model = self.model()
        points = torch.tensor([[0., 0., 0.], [.5, .5, .5], [1., 1., 1.],
                               [.17, .63, .88], [.35, .42, .59], [.9, .1, .5]])
        actual = model._fuse_evidence(list(points.T)).detach()
        aggregator = FuzzyEvidenceAggregator()
        expected = torch.tensor([aggregator(*row.tolist()) for row in points])
        torch.testing.assert_close(actual, expected)

    def test_grid_storage_is_grouped_by_three_consequents(self):
        model = self.model()
        saved = set()
        def pack(tensor):
            if tuple(tensor.shape) == (4, 201):
                saved.add(tensor.untyped_storage().data_ptr())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            model(torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(4).roll(1))
        self.assertLessEqual(len(saved), 6)

    def test_unsupported_fusion_and_incomplete_modalities_are_rejected(self):
        for mode in ('weighted', 'addition', 'concat'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.model(fusion_mode=mode)
        with self.assertRaisesRegex(ValueError, 'features for every entity'):
            self.model(visual_entity_to_index={0: 0})

    def test_training_checkpoint_and_reload(self):
        with tempfile.TemporaryDirectory(prefix='fuzzy_training_') as directory:
            root = Path(directory)
            entities = ['a', 'b', 'c', 'd']
            rows = [(entities[i], 'r', entities[(i + 1) % 4], 'train') for i in range(4)]
            rows += [('a', 'r', 'c', 'val'), ('b', 'r', 'd', 'test')]
            pd.DataFrame(rows, columns=['head', 'relation', 'tail', 'mode']).to_csv(root / 'triples.csv', index=False)
            vectors = np.eye(4, dtype=np.float32)
            for name in ('visual', 'textual'):
                with (root / (name + '.pkl')).open('wb') as stream:
                    pickle.dump(dict(zip(entities, vectors)), stream)
            checkpoint_path = root / 'model.pt'
            config = dict(dataset='wn9_img', model='TransEFuzzy', data_path=str(root / 'triples.csv'),
                          visual_features_path=str(root / 'visual.pkl'), textual_features_path=str(root / 'textual.pkl'),
                          embedding_dim=4, use_structural=True, use_visual=True, use_textual=True,
                          p_norm=1, epochs=1, batch_size=2, num_neg_samples=1, evaluate_every=1,
                          fuzzy_temperature=.7, save_path=str(checkpoint_path), lr=.02)
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            args = ['--config', str(config_path), '--no_cuda']
            with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['train'] + args):
                main()
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['model_name'], 'TransEFuzzy')
            self.assertEqual(checkpoint['training_config']['fuzzy_temperature'], .7)
            model, _, _ = helpers.get_model('TransEFuzzy', 4, 1, visual_features=torch.eye(4),
                                           textual_features=torch.eye(4),
                                           visual_entity_to_index=dict(enumerate(range(4))),
                                           textual_entity_to_index=dict(enumerate(range(4))),
                                           embedding_dim=4, use_visual=True, use_textual=True,
                                           fuzzy_temperature=.7)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.assertTrue(torch.isfinite(model(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))).all())
            self.assertEqual(parse_args(args + ['--fuzzy_temperature', '.9']).fuzzy_temperature, .9)
            # Fuzzy checkpoints must never be silently run through the base TransE runner.
            base_config = {**config, 'model': 'TransE'}
            config_path.write_text(yaml.safe_dump(base_config))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'trained as TransEFuzzy'):
                load_transe_fuzzy_environment(config_path, checkpoint_path, {}, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
