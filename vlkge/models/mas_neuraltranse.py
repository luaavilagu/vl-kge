"""
MASNeuralTransE new model inheriting from VLKGEBase
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlkge.models.vlkge import VLKGEBase


class MASNeuralTransE(VLKGEBase):
    """
    MASNeuralTransE: Neural Translating Embeddings for Modeling Multi-relational Data
    Scoring function: score = S_visual + S_textual
        where:
            S_visual  = margin - ||h_v + r - t_v||_p + (0.1 * Visual_ann)
            S_textual = margin - ||h_t + r - t_t||_p + (0.1 * Textual_ann) 
    
    Reference: Bordes et al. "Translating Embeddings for Modeling Multi-relational Data" (NeurIPS 2013)
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 visual_features=None, textual_features=None, relation_features=None,
                 visual_entity_to_index=None, textual_entity_to_index=None,
                 fusion_mode="average", 
                 use_structural=True, use_visual=False, use_textual=False,
                 freeze_visual=True, freeze_textual=True,
                 visual_proj=False, textual_proj=False,
                 shared_projection=None,
                 inductive=False,
                 modality_asymmetry=False,
                 normalize_before_fusion=False,
                 normalize_relations=True, p_norm=1,
                 raw_margin=12.0, device=torch.device('cpu')):
        """
        Args:
            ... (standard VLKGEBase args)
            relation_features: Optional pretrained relation features
            p_norm: Norm to use for distance (1 or 2)
            raw_margin: Margin value for scoring
        """
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
        
        # ==================== MASNeuralTransE-Specific: Relation Embeddings ====================
        if relation_features is not None:
            self.relation_embeddings = nn.Embedding.from_pretrained(relation_features, freeze=False)
            self.relation_proj = nn.Linear(self.relation_embeddings.embedding_dim, final_dim)
        else:
            self.relation_embeddings = nn.Embedding(num_relations, final_dim)
            self.relation_proj = None

        # A new NN added to create score for visual characteristics that approximates to TransE
        self.visual_scorer = nn.Sequential(
            nn.Linear(3*final_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

        # A new NN added to create score for textual characteristics that approximates to TransE
        self.textual_scorer = nn.Sequential(
            nn.Linear(3*final_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )   

        # Initialize parameters
        self.reset_parameters()        

    def reset_parameters(self):
        """Initialize MASNeuralTransE parameters with Xavier uniform"""
        # 1. Structural embeddings
        if self.use_structural:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # 2. Relation embeddings
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

        # 7. ANN for visual characteristics initialization
        for layer in self.visual_scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 8. ANN for textual characteristics initialization
        for layer in self.textual_scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def get_relation_representations(self, relation_ids):
        """Get relation embeddings"""
        rel_emb = self.relation_embeddings(relation_ids)
        
        if self.relation_proj is not None:
            rel_emb = self.relation_proj(rel_emb)
        
        # Optional normalization (helps with training stability) 
        if self.normalize_relations:
            rel_emb = F.normalize(rel_emb, p=2, dim=-1)
        
        return rel_emb

    def score_visual_evidence(self, head, relation, tail):
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

        visual_features = torch.cat(
            (visual_head, relation_emb, visual_tail),
            dim=-1
        )

        visual_ann_score = 0.1 * self.visual_scorer(
            visual_features
        ).squeeze(-1)

        return visual_ann_score + visual_transe_score

    def score_textual_evidence(self, head, relation, tail):
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

        textual_features = torch.cat(
            (textual_head, relation_emb, textual_tail),
            dim=-1
        )

        textual_ann_score = 0.1 * self.textual_scorer(
            textual_features
        ).squeeze(-1)

        return textual_ann_score + textual_transe_score

    def forward(self, head, relation, tail):
        """
        MASNeuralTransE forward pass: score = S_visual + S_textual
        where:
            S_visual  = margin - ||h_v + r - t_v||_p + (0.1 * Visual_ann)
            S_textual = margin - ||h_t + r - t_t||_p + (0.1 * Textual_ann) 
            
        Args:
            head: Head entity IDs (batch_size,)
            relation: Relation IDs (batch_size,)
            tail: Tail entity IDs (batch_size,)
            
        Returns:
            scores: Triple scores (batch_size,), higher is better
        """

        visual_score = self.score_visual_evidence(head, relation, tail)
        textual_score = self.score_textual_evidence(head, relation, tail)
        return visual_score + textual_score