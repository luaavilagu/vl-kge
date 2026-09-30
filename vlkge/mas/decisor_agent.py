import mesa

class DecisorAgent(mesa.Agent):
    """An agent which will make the decisions."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.goal = "Decide whether a candidate link is possible or not based on textual and visual characteristics"
        self.current_candidate = None
        self.textual_message = None
        self.visual_message = None
        self.decision_message = None
        self.last_message = None
        self.combined_score = None

    def perceive(self):
        # agent’s internal memory of what it perceived.
        self.current_candidate = self.model.current_candidate
        
        return self.current_candidate

    def reason(self):
        # decide whether the predicted link is feasible.
        for message in self.model.messages:
            if message["type"] == "textual_evidence":
                self.textual_message = message
            elif message["type"] == "visual_evidence":
                self.visual_message = message
            else:
                print("Unknown message received")

        self.combined_score = (self.visual_message["score"] + self.textual_message["score"]) / 2 

        if self.combined_score >= 0.5:
            self.decision_message = "accepted"
        else:
            self.decision_message = "rejected"

    def act(self):
       # put a message regarding with the candidate decision
        self.last_message = {
            "sender": "DecisorAgent",
            "type": "link_decision",
            "candidate": self.current_candidate,
            "decision": self.decision_message,
            "visual_score": self.visual_message["score"],
            "textual_score": self.textual_message["score"],
            "combined_score": self.combined_score
        }

        self.model.messages.append(self.last_message)

        return self.last_message

    def step(self):
        # execute Agent actions and return the candidate information and it's score.
        self.perceive()
        self.reason()

        return self.act() 
