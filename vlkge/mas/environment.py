import mesa

from vlkge.mas.visual_agent import VisualAgent
from vlkge.mas.textual_agent import TextualAgent
from vlkge.mas.decisor_agent import DecisorAgent

class KnowledgeGraphEnvironment(mesa.Model):
    """Shared environment for knowledge-graph evidence agents."""

    def __init__(self, visual_evidence_model, textual_evidence_model, seed=42):
        super().__init__(rng=seed)

        # Method use for the visual agent during reasoning
        self.visual_evidence_model = visual_evidence_model

        # Method use for the textual agent during reasoning
        self.textual_evidence_model = textual_evidence_model
                
        # Triple published by the SMA
        self.current_candidate = None
        
        # Shared social-message queue
        self.messages = []

        # Visual agent in the environment
        self.visual_agent = VisualAgent(self)

        # Textual agent in the environment
        self.textual_agent = TextualAgent(self)

        # Decisor agent in the environment
        self.decisor_agent = DecisorAgent(self)

    def publish_candidate(self, candidate):
        self.current_candidate = candidate
        self.messages.clear()

    def step(self):
        self.visual_agent.step()
        self.textual_agent.step()
        self.decisor_agent.step()
        return 

