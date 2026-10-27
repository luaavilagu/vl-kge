import mesa

class StructuralAgent(mesa.Agent):
    """An agent for structural analysis."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.goal = "Estimate structural evidence for a candidate link"
        self.current_candidate = None
        self.last_structural_score = None
        self.last_message = None

    def perceive(self):
        # agent’s internal memory of what it perceived.
        self.current_candidate = self.model.current_candidate
        
        return self.current_candidate

    def reason(self):
        # Delegate structural scoring to the evidence provider in the environment.
        self.last_structural_score = self.model.structural_evidence_model(
            self.current_candidate
        )

        return self.last_structural_score

    def act(self):
        # put a message regarding with the candidate information and it's score
        self.last_message = {
            "sender": "StructuralAgent",
            "type": "structural_evidence",
            "candidate": self.current_candidate,
            "score": self.last_structural_score
        }

        self.model.messages.append(self.last_message)

        return self.last_message

    def step(self):
        # execute Agent actions and return the candidate information and it's score.
        self.perceive()
        self.reason()

        return self.act() 
