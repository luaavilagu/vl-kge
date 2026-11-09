import torch
import torch.nn.functional as F


class EvidenceAdapter():
    """Translate candidate triples into scores for the evidence agents.

    Models with modality-specific scoring methods provide their own scores.
    A standard TransE model instead uses this adapter to score each enabled
    modality independently with the checkpoint's shared relation embedding.
    """

    def __init__(self, model, entity_to_id, relation_to_id, device):

        self.model = model
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.device = device

    def candidate_to_tensors(self, candidate):

        # Validate candidate is a dict 
        if not isinstance(candidate, dict):
            raise TypeError("Candidate must be a dictionary.")

        # Validate required fields in candidate
        required_fields = ("head", "relation", "tail")
        missing_fields = [
            field for field in required_fields
            if field not in candidate or candidate[field] is None
        ]

        # Throw an error if there is a missing field
        if missing_fields:
            raise ValueError(
                f"Candidate is missing required fields: {', '.join(missing_fields)}"
            )

        # Validate candidate["head"] exists in self.entity_to_id
        if candidate["head"] not in self.entity_to_id:
            raise ValueError(f"Unknown head entity: {candidate['head']}")

        # Validate candidate["tail"] exists in self.entity_to_id
        if candidate["tail"] not in self.entity_to_id:
            raise ValueError(f"Unknown tail entity: {candidate['tail']}")

        # Validate candidate["relation"] exists in self.relation_to_id
        if candidate["relation"] not in self.relation_to_id:
            raise ValueError(f"Unknown relation: {candidate['relation']}")

        # Get candidate values
        head_id = self.entity_to_id[candidate["head"]]
        relation_id = self.relation_to_id[candidate["relation"]]
        tail_id = self.entity_to_id[candidate["tail"]]

        # Create tensor for each candidate value
        head = torch.tensor([head_id], dtype=torch.long, device=self.device)
        relation = torch.tensor([relation_id], dtype=torch.long, device=self.device)
        tail = torch.tensor([tail_id], dtype=torch.long, device=self.device)
        
        return head, relation, tail

    def score_visual(self, candidate):
        return self._score_modality("visual", candidate)

    def score_textual(self, candidate):
        return self._score_modality("textual", candidate)

    def score_structural(self, candidate):
        return self._score_modality("structural", candidate)

    def _score_modality(self, modality, candidate):
        head, relation, tail = self.candidate_to_tensors(candidate)
        self.model.eval()
        model_scorer = getattr(self.model, f"score_{modality}_evidence", None)
        with torch.no_grad():
            if callable(model_scorer):
                score = model_scorer(head, relation, tail)
            else:
                score = self._score_transe_modality(
                    modality, head, relation, tail
                )

        return float(score.item())

    def _score_transe_modality(self, modality, head, relation, tail):
        """Score one TransE modality using the checkpoint's learned relation.

        This is a post-hoc modality-specific evidence score. The standard
        TransE training objective is still computed on fused entity vectors;
        this method does not claim that each modality was trained separately.
        """
        if not getattr(self.model, f"use_{modality}", False):
            raise ValueError(f"TransE checkpoint does not enable {modality} evidence.")

        embedding_getter = getattr(self.model, f"get_{modality}_embedding", None)
        if not callable(embedding_getter):
            raise TypeError(
                f"Model does not provide an embedding getter for {modality}."
            )
        head_embedding = embedding_getter(head)
        tail_embedding = embedding_getter(tail)
        if head_embedding is None or tail_embedding is None:
            raise ValueError(f"TransE returned no {modality} embeddings.")

        relation_embedding = self.model.get_relation_representations(relation)
        head_embedding = F.normalize(head_embedding, p=2, dim=-1)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=-1)
        return self.model.raw_margin - torch.norm(
            head_embedding + relation_embedding - tail_embedding,
            p=self.model.p_norm,
            dim=-1,
        )

    def fuse_scores(self, visual_score, textual_score, structural_score):
        """Adapt scalar agent evidence to MASGATransE's shared fusion policy.

        Model alpha must already be loaded from the GA result. The model
        returns a scalar tensor; agent messages carry its Python float value.
        The frozen scorer and no_grad context keep this inference-only.
        """
        self.model.eval()
        with torch.no_grad():
            score = self.model.fuse_evidence_scores(visual_score, textual_score, structural_score)
        return float(score.item())
