"""Cache frozen modality scores and evaluate filtered link-prediction ranks.

The cache contains scores, not embeddings. Changing alpha only recombines
two numbers per candidate, avoiding a model forward pass for every genome.
Head and tail filters are separate to avoid removing answers to the opposite
query direction. Equal scores are ordered by ascending entity ID, matching
a stable descending sort over an ascending candidate list.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


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


@dataclass
class ScoreCache:
    """Fixed scores and candidate masks for one validation or test split.

    scores has shape (queries, entities, 2), in visual/textual order.
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
        if self.scores.ndim != 3 or self.scores.shape[2] != 2:
            raise ValueError("scores must have shape (queries, entities, 2).")
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

    def metrics(self, alpha, ks=(1, 3, 10), query_batch_size=128):
        """Return filtered MRR, Hits@K, and per-relation metrics for one alpha.

        Blend before applying the Boolean filter. Storing -infinity inside
        modality scores would cause 0 * -infinity = NaN at alpha 0 or 1.
        Rank counting is equivalent to stable sorting but avoids sorting all
        entities. Float32 alpha matches the model's registered alpha buffer.
        Temporary fusion matrices are limited to query_batch_size rows.
        """
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1].")
        if query_batch_size < 1 or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError("Use positive query batch size and Hits@K values.")
        alpha = np.float32(alpha)
        candidate_ids = np.arange(self.scores.shape[1])
        ranks = np.empty(len(self.targets), dtype=np.int64)
        for start in range(0, len(ranks), query_batch_size):
            end = min(start + query_batch_size, len(ranks))
            block = self.scores[start:end]
            combined = alpha * block[:, :, 0] + (np.float32(1) - alpha) * block[:, :, 1]
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
    scores = np.empty((num_queries, model.num_entities, 2), dtype=np.float32)
    valid = np.ones((num_queries, model.num_entities), dtype=bool)
    targets = np.empty(num_queries, dtype=np.int64)
    relations = np.empty(num_queries, dtype=np.int64)
    device = model.alpha_tensor.device
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
                    scores[query, start:end, 0] = model.score_visual_evidence(h, r, t).cpu().numpy()
                    scores[query, start:end, 1] = model.score_textual_evidence(h, r, t).cpu().numpy()
                query += 1
                if progress is not None and (query % 50 == 0 or query == num_queries):
                    progress(query, num_queries)
    return ScoreCache(scores, valid, targets, relations)


def compare_searches(cache, budget, seed=42, grid_points=None, ks=(1, 3, 10)):
    """Compare equal/single-modality weights, random search, and grid search.

    Random search includes the same three reference weights as the GA and
    uses the GA's number of unique fitness evaluations as its budget. The
    grid uses that many evenly spaced points unless explicitly overridden.
    Winners are selected strictly by validation MRR; test data is not used.
    """
    import random

    if budget < 3 or (grid_points is not None and grid_points < 3):
        raise ValueError("Search comparisons require at least 3 evaluations.")
    references = {
        label: {"alpha": alpha, "validation": cache.metrics(alpha, ks=ks)}
        for label, alpha in (("textual_only", 0.0), ("equal_weights", 0.5), ("visual_only", 1.0))
    }
    generator = random.Random(seed)
    random_alphas = [0.0, 0.5, 1.0] + [generator.uniform(0, 1) for _ in range(budget - 3)]

    def best(alphas):
        """Evaluate each alpha and retain its full metrics for reporting."""
        winner = None
        for alpha in alphas:
            metrics = cache.metrics(float(alpha), ks=ks)
            if winner is None or metrics["MRR"] > winner["validation"]["MRR"]:
                winner = {"alpha": float(alpha), "validation": metrics}
        winner["evaluations"] = len(alphas)
        return winner

    references["random_search"] = best(random_alphas)
    references["grid_search"] = best(np.linspace(0, 1, grid_points or budget))
    return references
