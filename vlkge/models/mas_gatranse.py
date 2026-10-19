"""Separate TransE modality scores with an externally controlled fusion weight.

Relation vectors are learned once using the existing TransE training loop.
After loading that checkpoint, DEAP searches alpha while all model parameters
stay frozen. No ANN scoring layers or genetic operators belong in this model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlkge.models.vlkge import VLKGEBase

class MASGATransE(VLKGEBase):
    """
    Score modalities independently, then return alpha*S_visual+(1-alpha)*S_textual.

    Entity embeddings are L2-normalized within each modality. Both scores use
    the same learned relation translation, margin, and distance norm. Alpha is
    a scalar buffer, so checkpoints/device moves preserve it and optimizers
    exclude it. Training starts at alpha=0.5; GA search updates only alpha.
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 visual_features=None, textual_features=None, relation_features=None,
                 visual_entity_to_index=None, textual_entity_to_index=None,
                 fusion_mode="average", 
                 use_structural=False, use_visual=True, use_textual=True,
                 freeze_visual=True, freeze_textual=True,
                 visual_proj=False, textual_proj=False,
                 shared_projection=None,
                 inductive=False,
                 modality_asymmetry=False,
                 normalize_before_fusion=False,
                 normalize_relations=True, p_norm=1,
                 raw_margin=12.0, device=torch.device('cpu'),
                 alpha=0.5
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
            alpha: Finite visual weight in [0, 1], default 0.5. Textual weight
                is always derived as 1-alpha rather than stored independently.
            device: Initial device; the factory subsequently moves the model.
            Other options: Passed to VLKGEBase for feature loading/projection.
                Structural embeddings and embedding-level fusion are not part
                of this model's score; use the dedicated training configuration.
        """
        if use_structural or not (use_visual and use_textual):
            raise ValueError("MASGATransE requires visual/textual modalities and no structural embeddings.")
        if fusion_mode != "average":
            raise ValueError("Use fusion_mode='average'; alpha controls score-level fusion separately.")
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

        if not (alpha <= 1 and alpha >= 0):
            raise ValueError(f"Alpha is not between [0,1]. Got: {alpha  }")

        self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32),)

        # Initialize parameters
        self.reset_parameters()        

    def reset_parameters(self):
        """Initialize learned tables/projections, preserving supplied relations.

        This does not reset alpha. Supplied relation features, when explicitly
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

    def set_alpha(self, alpha):
        """Validate and fill the existing scalar buffer without tracking gradients.

        DEAP passes a Python number from an individual's single gene. In-place
        filling preserves the buffer's identity, dtype, device, and checkpoint
        registration. This method changes no embeddings or relation vectors.
        """
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError("The provided number must be between 0 and 1.")

        with torch.no_grad():
            self.alpha_tensor.fill_(alpha)

    def forward(self, head, relation, tail):
        """
        MASGATransE forward pass: score = (alpha) * S_visual + (1 - alpha) * S_textual
        where:
            S_visual  = margin - ||h_v + r - t_v||_p
            S_textual = margin - ||h_t + r - t_t||_p 
            alpha is a global weight between 0 and 1, defaulting to 0.5.
            
        Args:
            head: Head entity IDs (batch_size,)
            relation: Relation IDs (batch_size,)
            tail: Tail entity IDs (batch_size,)
            
        Returns:
            scores: Triple scores (batch_size,), higher is better
        """

        visual_score = self.score_visual_evidence(head, relation, tail)
        textual_score = self.score_textual_evidence(head, relation, tail)

        return (self.alpha_tensor) * visual_score + (1 - self.alpha_tensor) * textual_score
