import torch

class EvidenceAdapter():
    """Adapter to make possible inference translation"""

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
        head, relation, tail = self.candidate_to_tensors(candidate)

        self.model.eval()

        with torch.no_grad():
            score = self.model.score_visual_evidence(head, relation, tail)

        return float(score.item())

    def score_textual(self, candidate):
        head, relation, tail = self.candidate_to_tensors(candidate)

        self.model.eval()

        with torch.no_grad():
            score = self.model.score_textual_evidence(head, relation, tail)

        return float(score.item())


