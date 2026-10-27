import math

import mesa

class DecisorAgent(mesa.Agent):
    """Combine this candidate's evidence with the environment's fusion policy.

    The GA policy delegates to the frozen MASGATransE model, ensuring that
    agent aggregation agrees with the scoring model used to select alpha.
    Binary decisions are optional and require a supplied threshold.
    """

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.goal = "Decide whether a candidate link is possible or not based on textual, structural and visual characteristics"
        self.current_candidate = None
        self.textual_message = None
        self.visual_message = None
        self.structural_message = None
        self.decision_message = None
        self.last_message = None
        self.combined_score = None

    def perceive(self):
        # agent’s internal memory of what it perceived.
        self.current_candidate = self.model.current_candidate
        
        return self.current_candidate

    def reason(self):
        """Fuse fresh, matching evidence; optionally compare it to a threshold.

        Clear previous evidence before reading messages so a missing modality
        cannot reuse a score from a prior candidate. Ignore other candidates'
        messages. Raw TransE scores are not probabilities; no threshold means
        the result is explicitly unclassified, with its numeric score intact.
        """
        self.textual_message = None
        self.visual_message = None
        self.structural_message = None
        for message in self.model.messages:
            if message.get("candidate") != self.current_candidate:
                continue
            if message["type"] == "textual_evidence":
                self.textual_message = message
            elif message["type"] == "visual_evidence":
                self.visual_message = message
            elif message["type"] == "structural_evidence":
                self.structural_message = message
        if self.visual_message is None or self.textual_message is None or self.structural_message is None:
            raise ValueError("Visual, textual and structural evidence are required for the current candidate.")
        visual = float(self.visual_message["score"])
        textual = float(self.textual_message["score"])
        structural = float(self.structural_message["score"])
        if not math.isfinite(visual) or not math.isfinite(textual) or not math.isfinite(structural):
            raise ValueError("Evidence scores must be finite.")
        if self.model.score_fusion is None:
            self.combined_score = (visual + textual + structural) / 3
        else:
            self.combined_score = float(self.model.score_fusion(visual, textual, structural))
        if not math.isfinite(self.combined_score):
            raise ValueError("The fused score must be finite.")

        threshold = self.model.decision_threshold
        if threshold is None:
            self.decision_message = "unclassified"
        elif self.combined_score >= threshold:
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
            "structural_score": self.structural_message["score"],
            "combined_score": self.combined_score,
            "decision_threshold": self.model.decision_threshold,
        }

        self.model.messages.append(self.last_message)

        return self.last_message

    def step(self):
        # execute Agent actions and return the candidate information and it's score.
        self.perceive()
        self.reason()

        return self.act() 
