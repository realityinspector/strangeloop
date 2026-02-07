"""Behavior pattern controller.

Selects conversation patterns based on simulation phase,
relationship state, and ADSR intensity.
"""

import random

from strangeloop.knobs.envelope import EnvelopeConfig
from strangeloop.schemas.relationship import RelationshipState


# Pattern -> (min_intensity, max_intensity, min_trust)
PATTERN_PROFILES: dict[str, tuple[float, float, float]] = {
    "BANTER": (0.2, 0.7, 0.3),
    "DEBATE": (0.4, 0.9, 0.2),
    "ARGUMENT": (0.6, 1.0, 0.0),
    "CONFESSION": (0.5, 0.9, 0.5),
    "NEGOTIATION": (0.3, 0.8, 0.2),
    "STORYTELLING": (0.2, 0.6, 0.3),
    "INTERROGATION": (0.5, 1.0, 0.1),
}


class BehaviorController:
    """Selects conversation patterns based on simulation state."""

    def __init__(self, envelope: EnvelopeConfig, allowed_patterns: list[str] | None = None):
        self.envelope = envelope
        self.allowed = allowed_patterns or list(PATTERN_PROFILES.keys())

    def select_pattern(
        self,
        tick: int,
        total_ticks: int,
        relationship: RelationshipState | None = None,
        rng: random.Random | None = None,
    ) -> str:
        """Select a conversation pattern appropriate for the current state."""
        rng = rng or random.Random()
        intensity = self.envelope.intensity_at(tick / max(total_ticks, 1))

        candidates: list[tuple[str, float]] = []
        for pattern in self.allowed:
            profile = PATTERN_PROFILES.get(pattern)
            if not profile:
                continue
            min_i, max_i, min_trust = profile

            if not (min_i <= intensity <= max_i):
                continue

            trust = relationship.metrics.trust_level if relationship else 0.5
            if trust < min_trust:
                continue

            # Weight by how close intensity is to pattern's sweet spot
            center = (min_i + max_i) / 2
            fit = 1.0 - abs(intensity - center) / max(max_i - min_i, 0.1)
            candidates.append((pattern, fit))

        if not candidates:
            return "BANTER"

        # Weighted random selection
        total = sum(w for _, w in candidates)
        r = rng.random() * total
        cumulative = 0.0
        for pattern, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                return pattern

        return candidates[-1][0]
