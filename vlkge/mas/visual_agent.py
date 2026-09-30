import mesa

class VisualAgent(mesa.Agent):
    """An agent for visual analysis."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.goal = "Estimate visual evidence for a candidate link"
        self.current_candidate = None
        self.last_visual_score = None
        self.last_message = None

    def perceive(self):
        # agent’s internal memory of what it perceived.
        self.current_candidate = self.model.current_candidate
        
        return self.current_candidate

    def reason(self):
        # delegate visual scoring to a neural model provided by its environment.
        self.last_visual_score = self.model.visual_evidence_model(
            self.current_candidate
        )

        return self.last_visual_score

    def act(self):
        # put a message regarding with the candidate information and it's score
        self.last_message = {
            "sender": "VisualAgent",
            "type": "visual_evidence",
            "candidate": self.current_candidate,
            "score": self.last_visual_score
        }

        self.model.messages.append(self.last_message)

        return self.last_message

    def step(self):
        # execute Agent actions and return the candidate information and it's score.
        self.perceive()
        self.reason()

        return self.act() 
