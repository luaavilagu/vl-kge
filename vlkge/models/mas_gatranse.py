"""Three separate TransE scores with externally controlled simplex weights.

Relation vectors are learned once using the existing TransE training loop.
Structural entity embeddings and relations learn with equal thirds. DEAP then
searches relative contributions while every model parameter stays frozen.
No ANN scoring layers or genetic operators belong in this model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlkge.models.vlkge import VLKGEBase

MODALITIES = ("visual", "textual", "structural")
EQUAL_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


def validate_weights(weights):
    """Return three finite, nonnegative weights in the fixed modality order.

    Accept a sequence or a dictionary keyed by MODALITIES. Reject missing
    modalities, negative values, and sums outside a small roundoff tolerance.
    Normalize only that accepted roundoff so model, agents, and caches share
    the same interpretation. Do not silently repair arbitrary caller errors.
    """
    if isinstance(weights, dict):
        if set(weights) != set(MODALITIES):
            raise ValueError(f"Weights must name exactly {MODALITIES}.")
        weights = [weights[name] for name in MODALITIES]
    try:
        values = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected three numeric modality weights.") from error
    if len(values) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Expected three finite weights in [0, 1].")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Modality weights must sum to one.")
    return tuple(value / total for value in values)


class MASGATransE(VLKGEBase):
    """
    Return w_visual*S_visual + w_textual*S_textual + w_structural*S_structural.

    Each modality uses L2-normalized entities and the same relation vector.
    modality_weights is a three-element buffer saved in checkpoints and moved
    with the model, but excluded from gradient optimization. Its default is
    equal thirds. Structural entities and relations are trainable initially;
    the supplied visual/textual feature tables remain frozen by default.
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 visual_features=None, textual_features=None, relation_features=None,
                 visual_entity_to_index=None, textual_entity_to_index=None,
                 fusion_mode="average", 
                 use_structural=True, use_visual=True, use_textual=True,
                 freeze_visual=True, freeze_textual=True,
                 visual_proj=False, textual_proj=False,
                 shared_projection=None,
                 inductive=False,
                 modality_asymmetry=False,
                 normalize_before_fusion=False,
                 normalize_relations=True, p_norm=1,
                 raw_margin=12.0, device=torch.device('cpu'),
                 weights=None
                 ):
        """
        Args:
            num_entities, num_relations: Sizes of the dataset's ID vocabularies.
            embedding_dim: Dimension shared by modality and relation vectors.
            visual_features, textual_features: Supplied entity feature matrices.
            visual_entity_to_index, textual_entity_to_index: Entity ID to feature
                row mappings; both modalities are required for this experiment.
            relation_features: Optional initial relation features. The first
                experiment uses None and learns randomly initialized relations.
            p_norm: L1 or L2 distance, selected by 1 or 2.
            raw_margin: Constant added to negative translation distances.
            weights: Three nonnegative values summing to one, ordered visual,
                textual, structural; or a dictionary naming those modalities.
                None selects equal thirds. The GA changes this buffer only.
            device: Initial device; the factory subsequently moves the model.
            Other options: Passed to VLKGEBase for feature loading/projection.
                fusion_mode='average' retains base-class shape conventions;
                this class performs score-level rather than embedding fusion.
        """
        if not (use_structural and use_visual and use_textual):
            raise ValueError("MASGATransE requires structural, visual, and textual modalities.")
        if fusion_mode != "average":
            raise ValueError("Use fusion_mode='average'; modality weights control score-level fusion.")
        if p_norm not in (1, 2):
            raise ValueError("p_norm must be 1 or 2.")
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            visual_features=visual_features,
            textual_features=textual_features,
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
            device=device
        )
        
        self.normalize_relations = normalize_relations
        self.p_norm = p_norm
        self.raw_margin = nn.Parameter(torch.tensor(raw_margin), requires_grad=False)
        
        # Use final embedding_dim (which includes concat adjustment if needed)
        final_dim = self.embedding_dim
        
        # ==================== MASGATransE-Specific: Relation Embeddings ====================
        self._pretrained_relations = relation_features is not None
        if relation_features is not None:
            self.relation_embeddings = nn.Embedding.from_pretrained(relation_features, freeze=False)
            self.relation_proj = nn.Linear(self.relation_embeddings.embedding_dim, final_dim)
        else:
            self.relation_embeddings = nn.Embedding(num_relations, final_dim)
            self.relation_proj = None

        initial_weights = validate_weights(EQUAL_WEIGHTS if weights is None else weights)
        self.register_buffer("modality_weights", torch.tensor(initial_weights, dtype=torch.float32))

        # Initialize parameters
        self.reset_parameters()        

    def reset_parameters(self):
        """Initialize learned tables/projections, preserving supplied relations.

        This does not reset fusion weights. Supplied relation features, when explicitly
        used, retain their initial values rather than being overwritten.
        """
        # 1. Structural embeddings
        if self.use_structural:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # 2. Relation embeddings
        if not self._pretrained_relations:
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 3. Relation projection if exists
        if self.relation_proj is not None:
            nn.init.xavier_uniform_(self.relation_proj.weight)
        
        # 4. Visual projection layers
        if self.use_visual and hasattr(self, 'visual_linear'):
            if isinstance(self.visual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.visual_linear.weight)
        
        # 5. Textual projection layers
        if self.use_textual and hasattr(self, 'textual_linear'):
            if isinstance(self.textual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.textual_linear.weight)
        
        # 6. Shared projection if exists
        if len(self.shared_projection) > 0 and hasattr(self, 'unified_projection'):
            if isinstance(self.unified_projection, nn.Linear):
                nn.init.xavier_uniform_(self.unified_projection.weight)
    
    def get_relation_representations(self, relation_ids):
        """Look up trainable relation vectors, project and optionally normalize."""
        rel_emb = self.relation_embeddings(relation_ids)
        
        if self.relation_proj is not None:
            rel_emb = self.relation_proj(rel_emb)
        
        # Optional normalization (helps with training stability) 
        if self.normalize_relations:
            rel_emb = F.normalize(rel_emb, p=2, dim=-1)
        
        return rel_emb

    def score_visual_evidence(self, head, relation, tail):
        """Return margin minus visual TransE distance, one score per triple.

        Keep gradients enabled during training so the shared relations (and
        any configured dimension projections) can be learned from the loss.
        """
        visual_head = self.get_visual_embedding(head)
        visual_tail = self.get_visual_embedding(tail)
        relation_emb = self.get_relation_representations(relation)

        visual_head = F.normalize(visual_head, p=2, dim=-1)
        visual_tail = F.normalize(visual_tail, p=2, dim=-1)

        visual_transe_score = self.raw_margin - torch.norm(
            visual_head + relation_emb - visual_tail, 
            p=self.p_norm,
            dim=-1
        )

        return visual_transe_score

    def score_textual_evidence(self, head, relation, tail):
        """Return margin minus textual TransE distance, one score per triple."""
        textual_head = self.get_textual_embedding(head)
        textual_tail = self.get_textual_embedding(tail)
        relation_emb = self.get_relation_representations(relation)

        textual_head = F.normalize(textual_head, p=2, dim=-1)
        textual_tail = F.normalize(textual_tail, p=2, dim=-1)

        textual_transe_score = self.raw_margin - torch.norm(
            textual_head + relation_emb - textual_tail, 
            p=self.p_norm,
            dim=-1
        )

        return textual_transe_score

    def score_structural_evidence(self, head, relation, tail):
        """Return structural TransE evidence with no neural scoring head.

        The structural entity table and shared relations receive gradients
        during checkpoint training and are frozen before genetic search.
        """
        head_emb = F.normalize(self.get_structural_embedding(head), p=2, dim=-1)
        tail_emb = F.normalize(self.get_structural_embedding(tail), p=2, dim=-1)
        relation_emb = self.get_relation_representations(relation)
        return self.raw_margin - torch.norm(head_emb + relation_emb - tail_emb,
                                            p=self.p_norm, dim=-1)

    def set_weights(self, weights):
        """Validate and copy three weights in place without tracking gradients.

        Preserve buffer identity/device/dtype and leave every model parameter
        unchanged. DEAP individuals and named weight dictionaries are accepted.
        """
        values = validate_weights(weights)
        with torch.no_grad():
            self.modality_weights.copy_(self.modality_weights.new_tensor(values))

    def fuse_evidence_scores(self, visual_score, textual_score, structural_score):
        """Share the same differentiable three-score fusion with the agents."""
        scores = [torch.as_tensor(value, dtype=self.modality_weights.dtype,
                                  device=self.modality_weights.device)
                  for value in (visual_score, textual_score, structural_score)]
        # Keep this operation order identical to the NumPy ranking cache.
        return (self.modality_weights[0] * scores[0] + self.modality_weights[1] * scores[1]
                + self.modality_weights[2] * scores[2])

    def forward(self, head, relation, tail):
        """
        Fuse visual, textual, and structural TransE scores using the buffer.
        Each score is margin minus its modality's translation distance. The
        default weights are equal thirds and remain fixed during training.
            
        Args:
            head: Head entity IDs (batch_size,)
            relation: Relation IDs (batch_size,)
            tail: Tail entity IDs (batch_size,)
            
        Returns:
            scores: Triple scores (batch_size,), higher is better
        """

        visual_score = self.score_visual_evidence(head, relation, tail)
        textual_score = self.score_textual_evidence(head, relation, tail)
        structural_score = self.score_structural_evidence(head, relation, tail)
        return self.fuse_evidence_scores(visual_score, textual_score, structural_score)
