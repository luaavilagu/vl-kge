"""Mamdani fuzzy aggregation for visual, textual, and structural evidence."""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Tuple


_MODALITIES = ("visual", "textual", "structural")
_TERMS = ("low", "medium", "high")


@dataclass(frozen=True)
class FuzzyResult:
    """Defuzzified confidence and its strongest linguistic category."""

    score: float
    label: str


class FuzzyEvidenceAggregator:
    """Aggregate three calibrated evidence scores using Mamdani inference.

    Inputs must lie in [0, 1] after normalization. ``score_ranges`` can map
    raw modality scores to that interval; specify validation-derived bounds
    for every modality whose scores are not already normalized.
    """

    def __init__(self, score_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
                 resolution: int = 201):
        if type(resolution) is not int or resolution < 3:
            raise ValueError("resolution must be an integer of at least 3.")
        if score_ranges is None:
            score_ranges = {}
        unknown = set(score_ranges) - set(_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modality names in score_ranges: {sorted(unknown)}")

        self.score_ranges = {}
        for modality in _MODALITIES:
            bounds = score_ranges.get(modality, (0.0, 1.0))
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"{modality} score range must contain a minimum and maximum.")
            minimum, maximum = map(float, bounds)
            if (not math.isfinite(minimum) or not math.isfinite(maximum)
                    or minimum >= maximum):
                raise ValueError(f"{modality} score range must be finite and increasing.")
            self.score_ranges[modality] = (minimum, maximum)
        self.resolution = resolution

    @staticmethod
    def _membership(value, term):
        """Evaluate low/medium/high trapezoidal or triangular membership."""
        if term == "low":
            if value <= 0.25:
                return 1.0
            if value >= 0.5:
                return 0.0
            return (0.5 - value) / 0.25
        if term == "medium":
            if value <= 0.25 or value >= 0.75:
                return 0.0
            if value <= 0.5:
                return (value - 0.25) / 0.25
            return (0.75 - value) / 0.25
        if value <= 0.5:
            return 0.0
        if value >= 0.75:
            return 1.0
        return (value - 0.5) / 0.25

    @staticmethod
    def _consequent(terms):
        """Map the three linguistic inputs to a consensus output category."""
        strength = sum({"low": 0.0, "medium": 0.5, "high": 1.0}[term]
                       for term in terms)
        if strength <= 0.75:
            return "low"
        if strength >= 2.25:
            return "high"
        return "medium"

    def _normalize(self, modality, score):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"{modality} evidence score must be finite.")
        minimum, maximum = self.score_ranges[modality]
        if not minimum <= score <= maximum:
            raise ValueError(
                f"{modality} evidence score {score} is outside calibrated range "
                f"[{minimum}, {maximum}]."
            )
        return (score - minimum) / (maximum - minimum)

    def evaluate(self, visual, textual, structural):
        """Return a [0, 1] fuzzy confidence score and low/medium/high label."""
        inputs = {
            "visual": self._normalize("visual", visual),
            "textual": self._normalize("textual", textual),
            "structural": self._normalize("structural", structural),
        }
        memberships = {
            modality: {
                term: self._membership(value, term)
                for term in _TERMS
            }
            for modality, value in inputs.items()
        }
        output = [0.0] * self.resolution
        for visual_term in _TERMS:
            for textual_term in _TERMS:
                for structural_term in _TERMS:
                    terms = (visual_term, textual_term, structural_term)
                    firing = min(
                        memberships["visual"][visual_term],
                        memberships["textual"][textual_term],
                        memberships["structural"][structural_term],
                    )
                    if firing == 0:
                        continue
                    consequent = self._consequent(terms)
                    for index in range(self.resolution):
                        value = index / (self.resolution - 1)
                        output_membership = self._membership(value, consequent)
                        output[index] = max(output[index], min(firing, output_membership))

        denominator = sum(output)
        if denominator == 0:
            raise RuntimeError("Fuzzy rule base produced no output membership.")
        score = sum(
            (index / (self.resolution - 1)) * membership
            for index, membership in enumerate(output)
        ) / denominator
        label = max(_TERMS, key=lambda term: self._membership(score, term))
        return FuzzyResult(score=score, label=label)

    def __call__(self, visual, textual, structural):
        """Return only the fuzzy score for callers expecting a fusion function."""
        return self.evaluate(visual, textual, structural).score
