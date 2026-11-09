"""Cache frozen modality scores and evaluate filtered link-prediction ranks.

The cache contains scores, not embeddings. Each weight vector recombines
three numbers per candidate without retraining or rescoring embeddings.
Head and tail filters are separate to avoid removing answers to the opposite
query direction. Equal scores are ordered by ascending entity ID, matching
a stable descending sort over an ascending candidate list.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from vlkge.models.mas_gatranse import MODALITIES, validate_weights
from vlkge.ga.modality_fusion_ga import reference_weights, sample_weights, named_weights


def directional_filters(known_triples):
    """Build (head, relation)->tails and (tail, relation)->heads separately.

    known_triples has columns [head_id, relation_id, tail_id]. Using all
    observed splits here is the standard all-known filtered protocol: held-out
    answers are used only to exclude other true answers, never as loss labels
    or fitness targets. Each evaluated query's own target is always retained.
    """
    tails, heads = defaultdict(set), defaultdict(set)
    for head, relation, tail in np.asarray(known_triples, dtype=np.int64):
        tails[(int(head), int(relation))].add(int(tail))
        heads[(int(tail), int(relation))].add(int(head))
    return tails, heads


def known_triples_for_split(loader, split):
    """Use train+validation positives for search; include test only at final test.

    Keep the full validation set in the filters even for an explicit smoke
    subset. This changes the old GA all-split validation protocol; recompute
    comparison baselines under this protocol rather than reuse old numbers.
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'.")
    frame = loader.data
    if split == "val":
        frame = frame[frame["mode"].isin(("train", "val"))]
    return frame[["hid", "rid", "tid"]].to_numpy(dtype=np.int64)


@dataclass
class ScoreCache:
    """Fixed scores and candidate masks for one validation or test split.

    scores has shape (queries, entities, 3), ordered visual/textual/structural.
    valid has shape (queries, entities). targets and relations identify each
    query's correct answer and relation. Arrays remain on CPU, while model
    scoring can run on CPU or GPU. Full WN9-IMG caches require RAM; candidate
    scoring is chunked so the feature-distance intermediates stay bounded.
    """

    scores: np.ndarray
    valid: np.ndarray
    targets: np.ndarray
    relations: np.ndarray

    def __post_init__(self):
        """Reject malformed or nonfinite caches before optimization begins."""
        if self.scores.ndim != 3 or self.scores.shape[2] != 3:
            raise ValueError("scores must have shape (queries, entities, 3).")
        queries, entities, _ = self.scores.shape
        if not queries or not entities:
            raise ValueError("Cannot evaluate an empty split or candidate set.")
        if self.valid.shape != (queries, entities) or self.valid.dtype != np.bool_:
            raise ValueError("valid must be a Boolean query/candidate mask.")
        if self.targets.shape != (queries,) or self.relations.shape != (queries,):
            raise ValueError("Each query needs one target and relation ID.")
        if np.any(self.targets < 0) or np.any(self.targets >= entities):
            raise ValueError("Target IDs are outside the candidate set.")
        if not np.isfinite(self.scores).all():
            raise ValueError("Modality scores must be finite before filtering.")
        if not self.valid[np.arange(queries), self.targets].all():
            raise ValueError("Filtering must retain every query's own target.")

    def metrics(self, weights, ks=(1, 3, 10), query_batch_size=128):
        """Return filtered MRR, Hits@K, and per-relation metrics for one vector.

        Blend before applying the Boolean filter. Storing -infinity inside
        modality scores would cause 0 * -infinity = NaN for single modalities.
        Rank counting is equivalent to stable sorting but avoids sorting all
        entities. Float32 weights match the model's registered vector buffer.
        Temporary fusion matrices are limited to query_batch_size rows.
        """
        weights = np.asarray(validate_weights(weights), dtype=np.float32)
        if query_batch_size < 1 or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError("Use positive query batch size and Hits@K values.")
        candidate_ids = np.arange(self.scores.shape[1])
        ranks = np.empty(len(self.targets), dtype=np.int64)
        for start in range(0, len(ranks), query_batch_size):
            end = min(start + query_batch_size, len(ranks))
            block = self.scores[start:end]
            combined = (weights[0] * block[:, :, 0] + weights[1] * block[:, :, 1]
                        + weights[2] * block[:, :, 2])
            targets = self.targets[start:end]
            gold = combined[np.arange(end - start), targets, None]
            precedes = (combined > gold) | (
                (combined == gold) & (candidate_ids[None, :] < targets[:, None])
            )
            ranks[start:end] = 1 + np.count_nonzero(precedes & self.valid[start:end], axis=1)

        def summarize(values):
            """Use float64 reciprocals for consistent aggregate reporting."""
            return {
                "MRR": float(np.mean(1.0 / values)),
                **{f"Hits@{k}": float(np.mean(values <= k)) for k in ks},
                "queries": int(len(values)),
            }

        result = summarize(ranks)
        result["per_relation"] = {
            str(int(relation)): summarize(ranks[self.relations == relation])
            for relation in np.unique(self.relations)
        }
        return result


def build_score_cache(model, triples, known_triples, bidirectional=True,
                      candidate_batch_size=1024, progress=None):
    """Score every candidate entity once per query using the frozen model.

    triples supplies evaluation targets; known_triples supplies only filters.
    Tail queries are (h,r,?), and optional head queries are (?,r,t). Every
    entity is a candidate. This first implementation targets dense WN9-IMG
    features; sparse/per-relation candidate policies are intentionally outside
    its scope. The caller must keep the model unchanged while reusing a cache.
    """
    triples = np.asarray(triples, dtype=np.int64)
    known_triples = np.asarray(known_triples, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
        raise ValueError("Evaluation triples must be a nonempty (N, 3) array.")
    if known_triples.ndim != 2 or known_triples.shape[1] != 3:
        raise ValueError("Known triples must have shape (N, 3).")
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive.")
    tails_filter, heads_filter = directional_filters(known_triples)
    num_queries = len(triples) * (2 if bidirectional else 1)
    scores = np.empty((num_queries, model.num_entities, 3), dtype=np.float32)
    valid = np.ones((num_queries, model.num_entities), dtype=bool)
    targets = np.empty(num_queries, dtype=np.int64)
    relations = np.empty(num_queries, dtype=np.int64)
    device = model.modality_weights.device
    model.eval()
    query = 0
    with torch.no_grad():
        for head, relation, tail in triples:
            directions = ("tail", "head") if bidirectional else ("tail",)
            for direction in directions:
                target = int(tail if direction == "tail" else head)
                excluded = (tails_filter.get((int(head), int(relation)), set())
                            if direction == "tail" else
                            heads_filter.get((int(tail), int(relation)), set()))
                if excluded:
                    valid[query, list(excluded)] = False
                valid[query, target] = True
                targets[query] = target
                relations[query] = relation
                for start in range(0, model.num_entities, candidate_batch_size):
                    end = min(start + candidate_batch_size, model.num_entities)
                    candidates = torch.arange(start, end, device=device)
                    h = candidates if direction == "head" else torch.full_like(candidates, int(head))
                    t = candidates if direction == "tail" else torch.full_like(candidates, int(tail))
                    r = torch.full_like(candidates, int(relation))
                    for index, name in enumerate(MODALITIES):
                        scorer = getattr(model, f"score_{name}_evidence")
                        scores[query, start:end, index] = scorer(h, r, t).cpu().numpy()
                query += 1
                if progress is not None and (query % 50 == 0 or query == num_queries):
                    progress(query, num_queries)
    return ScoreCache(scores, valid, targets, relations)


def compare_searches(cache, budget, seed=42, grid_points=None, ks=(1, 3, 10)):
    """Compare equal/single-modality weights, random search, and grid search.

    Random search includes the same four references as the GA and uses its
    unique-evaluation budget. The simplex grid uses the largest complete
    triangular lattice within grid_points (or that budget). Report its actual
    count, which can be smaller than the cap. A 0.05 grid has 231 candidates.
    Winners are selected strictly by validation MRR; test data is not used.
    """
    import random

    if budget < 4 or (grid_points is not None and grid_points < 3):
        raise ValueError("Use a search budget of at least 4 and grid cap of at least 3.")
    references = {
        label: {"weights": named_weights(weights), "validation": cache.metrics(weights, ks=ks)}
        for label, weights in zip(("equal_weights", "visual_only", "textual_only", "structural_only"),
                                  reference_weights())
    }
    generator = random.Random(seed)
    random_vectors = reference_weights() + [sample_weights(generator) for _ in range(budget - 4)]

    def best(vectors):
        """Evaluate every vector and retain the validation winner's metrics."""
        winner = None
        for weights in vectors:
            metrics = cache.metrics(weights, ks=ks)
            if winner is None or metrics["MRR"] > winner["validation"]["MRR"]:
                winner = {"weights": named_weights(weights), "validation": metrics}
        winner["evaluations"] = len(vectors)
        return winner

    references["random_search"] = best(random_vectors)
    cap = grid_points or budget
    divisions = 1
    while (divisions + 2) * (divisions + 3) // 2 <= cap:
        divisions += 1
    vectors = [(i / divisions, j / divisions, (divisions - i - j) / divisions)
               for i in range(divisions + 1) for j in range(divisions - i + 1)]
    references["grid_search"] = best(vectors)
    references["grid_search"].update({"divisions": divisions, "evaluation_cap": cap})
    return references
