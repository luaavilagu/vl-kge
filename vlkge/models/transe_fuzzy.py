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
        if fusion_mode == "concat":
            raise ValueError("TransEFuzzy does not support concat fusion.")
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
            evidence.append(
                torch.sigmoid(
                    (self.fuzzy_distance_thresholds[index] - distance)
                    / self.fuzzy_temperature
                )
            )

        terms = ("low", "medium", "high")
        memberships = [
            {term: self._membership(value, term) for term in terms}
            for value in evidence
        ]
        grid = torch.linspace(
            0.0, 1.0, 201, device=head.device, dtype=evidence[0].dtype
        )
        aggregate = torch.zeros(
            (head.shape[0], grid.shape[0]),
            device=head.device,
            dtype=evidence[0].dtype,
        )
        term_values = {"low": 0.0, "medium": 0.5, "high": 1.0}
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
                    output_membership = self._membership(
                        grid.unsqueeze(0), consequent
                    )
                    aggregate = torch.maximum(
                        aggregate,
                        torch.minimum(firing.unsqueeze(1), output_membership),
                    )

        denominator = aggregate.sum(dim=1).clamp_min(
            torch.finfo(aggregate.dtype).eps
        )
        fuzzy_value = (aggregate * grid.unsqueeze(0)).sum(dim=1) / denominator
        return self.raw_margin * (2.0 * fuzzy_value - 1.0)
