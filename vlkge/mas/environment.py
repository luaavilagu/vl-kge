import math

import mesa

from vlkge.mas.visual_agent import VisualAgent
from vlkge.mas.textual_agent import TextualAgent
from vlkge.mas.structural_agent import StructuralAgent
from vlkge.mas.decisor_agent import DecisorAgent

class KnowledgeGraphEnvironment(mesa.Model):
    """Shared environment for knowledge-graph evidence agents."""

    def __init__(self, visual_evidence_model, textual_evidence_model, structural_evidence_model, seed=42,
                 score_fusion=None, decision_threshold=0.5):
        """Connect evidence providers and an optional decision fusion policy.

        score_fusion accepts visual/textual scalar scores and returns a fused
        score. MASGATransE supplies its trained model's GA-weighted policy.
        Omitting it preserves the original equal-weight policy for ANN runs.

        decision_threshold=None reports a score without claiming a binary
        classification. A finite explicit threshold enables accepted/rejected
        output. The legacy default remains 0.5 for existing callers; the GA
        runner explicitly chooses None because MRR does not calibrate a
        classification threshold.
        """
        if score_fusion is not None and not callable(score_fusion):
            raise TypeError("score_fusion must be callable or None.")
        if decision_threshold is not None and not math.isfinite(decision_threshold):
            raise ValueError("decision_threshold must be finite or None.")
        super().__init__(rng=seed)

        # Method use for the visual agent during reasoning
        self.visual_evidence_model = visual_evidence_model

        # Method use for the textual agent during reasoning
        self.textual_evidence_model = textual_evidence_model

        # Method use for the structural agent during reasoning
        self.structural_evidence_model = structural_evidence_model

        self.score_fusion = score_fusion
        self.decision_threshold = decision_threshold
                
        # Triple published by the SMA
        self.current_candidate = None
        
        # Shared social-message queue
        self.messages = []

        # Visual agent in the environment
        self.visual_agent = VisualAgent(self)

        # Textual agent in the environment
        self.textual_agent = TextualAgent(self)

        # Structural agent in the environment
        self.structural_agent = StructuralAgent(self)

        # Decisor agent in the environment
        self.decisor_agent = DecisorAgent(self)

    def publish_candidate(self, candidate):
        self.current_candidate = candidate
        self.messages.clear()

    def step(self):
        self.visual_agent.step()
        self.textual_agent.step()
        self.structural_agent.step()
        self.decisor_agent.step()
        return 

