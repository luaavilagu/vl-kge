"""TransE variant with differentiable fuzzy modality fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlkge.models.transe import TransE


class TransEFuzzy(TransE):
    """Train TransE embeddings using fuzzy fusion of modality distances."""

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim,
        visual_features=None,
        textual_features=None,
        relation_features=None,
        visual_entity_to_index=None,
        textual_entity_to_index=None,
        fusion_mode="average",
        use_structural=True,
        use_visual=False,
        use_textual=False,
        freeze_visual=True,
        freeze_textual=True,
        visual_proj=False,
        textual_proj=False,
        shared_projection=None,
        inductive=False,
        modality_asymmetry=False,
        normalize_before_fusion=False,
        normalize_relations=True,
        p_norm=1,
        fuzzy_temperature=1.0,
        raw_margin=12.0,
        device=torch.device("cpu"),
    ):
        if not (use_structural and use_visual and use_textual):
            raise ValueError(
                "TransEFuzzy requires structural, visual, and textual modalities."
            )
        if fusion_mode != "average":
            raise ValueError("TransEFuzzy does not support concat, weighted, or addition fusion; use average.")
        if shared_projection:
            raise ValueError("TransEFuzzy does not support shared_projection.")
        if inductive or modality_asymmetry:
            raise ValueError("TransEFuzzy requires complete modalities and transductive entities.")
        for name, mapping in (("visual", visual_entity_to_index), ("textual", textual_entity_to_index)):
            if mapping is None or set(mapping) != set(range(num_entities)):
                raise ValueError(f"TransEFuzzy requires {name} features for every entity.")
        if not math.isfinite(fuzzy_temperature) or fuzzy_temperature <= 0:
            raise ValueError(
                "fuzzy_temperature must be finite and greater than zero."
            )

        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            visual_features=visual_features,
            textual_features=textual_features,
            relation_features=relation_features,
            visual_entity_to_index=visual_entity_to_index,
            textual_entity_to_index=textual_entity_to_index,
            fusion_mode=fusion_mode,
            use_structural=use_structural,
            use_visual=use_visual,
            use_textual=use_textual,
            freeze_visual=freeze_visual,
            freeze_textual=freeze_textual,
            visual_proj=visual_proj,
            textual_proj=textual_proj,
            shared_projection=shared_projection,
            inductive=inductive,
            modality_asymmetry=modality_asymmetry,
            normalize_before_fusion=normalize_before_fusion,
            normalize_relations=normalize_relations,
            p_norm=p_norm,
            raw_margin=raw_margin,
            device=device,
        )
        self.fuzzy_temperature = fuzzy_temperature
        self.fuzzy_distance_thresholds = nn.Parameter(torch.ones(3))

    @staticmethod
    def _membership(value, term):
        if term == "low":
            return ((0.5 - value) / 0.25).clamp(0.0, 1.0)
        if term == "medium":
            return (1.0 - torch.abs(value - 0.5) / 0.25).clamp(0.0, 1.0)
        if term == "high":
            return ((value - 0.5) / 0.25).clamp(0.0, 1.0)
        raise ValueError(f"Unknown fuzzy membership term: {term}")

    def forward(self, head, relation, tail):
        relation_emb = self.get_relation_representations(relation)
        entity_ids = (head, tail)
        embedding_getters = (
            self.get_structural_embedding,
            self.get_visual_embedding,
            self.get_textual_embedding,
        )
        evidence = []
        logits = []
        for index, get_embedding in enumerate(embedding_getters):
            head_emb = F.normalize(get_embedding(entity_ids[0]), p=2, dim=-1)
            tail_emb = F.normalize(get_embedding(entity_ids[1]), p=2, dim=-1)
            distance = torch.norm(
                head_emb + relation_emb - tail_emb,
                p=self.p_norm,
                dim=-1,
            )
            if self.p_norm == 1:
                distance = distance / math.sqrt(self.embedding_dim)
            logit = (self.fuzzy_distance_thresholds[index] - distance) / self.fuzzy_temperature
            logits.append(logit)
            evidence.append(torch.sigmoid(logit))

        # The continuous residual preserves gradients and ranking differences
        # even when all linguistic memberships lie on constant plateaus.
        fuzzy_value = self._fuse_evidence(evidence)
        return self.raw_margin * (2.0 * fuzzy_value - 1.0) + torch.stack(logits).mean(dim=0)

    def _fuse_evidence(self, evidence):
        """Exact Mamdani centroid with only three grid-sized rule outputs.

        max_i min(a_i, B) == min(max_i a_i, B), so rules can be
        grouped by consequent before broadcasting onto the output grid.
        """

        terms = ("low", "medium", "high")
        memberships = [
            {term: self._membership(value, term) for term in terms}
            for value in evidence
        ]
        grid = torch.linspace(
            0.0, 1.0, 201, device=evidence[0].device, dtype=evidence[0].dtype
        )
        aggregate = torch.zeros(
            (evidence[0].shape[0], grid.shape[0]),
            device=evidence[0].device,
            dtype=evidence[0].dtype,
        )
        term_values = {"low": 0.0, "medium": 0.5, "high": 1.0}
        strengths = {term: torch.zeros_like(evidence[0]) for term in terms}
        for structural_term in terms:
            for visual_term in terms:
                for textual_term in terms:
                    labels = (structural_term, visual_term, textual_term)
                    firing = torch.minimum(
                        torch.minimum(
                            memberships[0][structural_term],
                            memberships[1][visual_term],
                        ),
                        memberships[2][textual_term],
                    )
                    strength = sum(term_values[label] for label in labels)
                    if strength <= 0.75:
                        consequent = "low"
                    elif strength >= 2.25:
                        consequent = "high"
                    else:
                        consequent = "medium"
                    strengths[consequent] = torch.maximum(strengths[consequent], firing)
        for consequent in terms:
            output_membership = self._membership(grid.unsqueeze(0), consequent)
            aggregate = torch.maximum(
                aggregate,
                torch.minimum(strengths[consequent].unsqueeze(1), output_membership),
            )

        denominator = aggregate.sum(dim=1).clamp_min(
            torch.finfo(aggregate.dtype).eps
        )
        fuzzy_value = (aggregate * grid.unsqueeze(0)).sum(dim=1) / denominator
        return fuzzy_value
