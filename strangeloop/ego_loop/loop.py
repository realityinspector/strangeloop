"""Ego Loop: inner monologue, goals, social monitoring.

One LLM call per synth per tick produces:
- Inner monologue (what the synth thinks but doesn't say)
- Updated goals (what the synth wants to achieve)
- Social monitoring notes (what they've noticed about others)

Extroversion gates how much social monitoring occurs.
"""

from dataclasses import dataclass

from strangeloop.gpt_genius.client import LLMClient
from strangeloop.schemas.synth import SynthProfile
from strangeloop.schemas.tensor import CognitiveTensor
from strangeloop.schemas.event import SimEvent


@dataclass
class EgoState:
    """Output of one ego loop tick."""
    inner_monologue: str
    goals: list[str]
    social_observations: list[str]
    updated_cognitive: CognitiveTensor


EGO_SYSTEM = """You are the inner mind of a character in a social simulation.
Given the character's profile, current state, and recent events,
produce their internal thoughts as JSON:
{
  "inner_monologue": "<what they're thinking but won't say aloud, 1-3 sentences>",
  "goals": ["<goal 1>", "<goal 2>", "<goal 3>"],
  "social_observations": ["<observation about another person>", ...],
  "emotional_valence": <-1.0 to 1.0, updated mood>,
  "emotional_arousal": <0.0 to 1.0, updated arousal>,
  "energy_delta": <-20 to 20, energy change>
}

Goals should be concrete and achievable within a social conversation.
Social observations should reflect what they've noticed about others' behavior.
The inner monologue reveals their true feelings, insecurities, strategies."""


class EgoLoop:
    """Runs inner monologue for each synth each tick."""

    def __init__(self, client: LLMClient):
        self.client = client

    def tick(
        self,
        synth: SynthProfile,
        cognitive: CognitiveTensor,
        events: list[SimEvent],
        recent_dialog_summary: str = "",
        tick_number: int = 0,
    ) -> EgoState:
        """Run one ego loop iteration for a synth."""
        # Build context
        event_text = ""
        if events:
            event_text = "Recent events:\n" + "\n".join(
                f"- {e.title}: {e.description[:80]}" for e in events[:3]
            )

        # Extroversion gates how much social monitoring detail we request
        social_depth = "detailed" if synth.psychographics.big_five.extraversion > 0.6 else "brief"

        user_prompt = (
            f"Character: {synth.name}, {synth.demographics.age}yo {synth.demographics.occupation}\n"
            f"Mood: valence={cognitive.emotional_valence:.1f}, arousal={cognitive.emotional_arousal:.1f}\n"
            f"Energy: {cognitive.energy_budget:.0f}/100\n"
            f"Current goals: {', '.join(cognitive.knowledge_state[:3]) or 'none yet'}\n"
            f"Motivations: {', '.join(synth.backstory.motivations[:3])}\n"
            f"Fears: {', '.join(synth.backstory.fears[:2])}\n"
            f"{event_text}\n"
            f"Recent conversation: {recent_dialog_summary[:200] or 'none'}\n"
            f"Social monitoring depth: {social_depth}\n"
            f"Tick: {tick_number}"
        )

        result = self.client.call(
            messages=[
                {"role": "system", "content": EGO_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=300,
            json_mode=True,
            fallback_response={
                "inner_monologue": "Just going about my day...",
                "goals": ["get through the day"],
                "social_observations": [],
                "emotional_valence": cognitive.emotional_valence,
                "emotional_arousal": cognitive.emotional_arousal,
                "energy_delta": 0,
            },
        )

        return self._parse_result(result, cognitive)

    def _parse_result(self, result: dict, current: CognitiveTensor) -> EgoState:
        """Parse ego loop LLM result."""
        if isinstance(result, dict) and "error" not in result:
            updated = CognitiveTensor(
                knowledge_state=current.knowledge_state,
                emotional_valence=max(-1.0, min(1.0, result.get("emotional_valence", current.emotional_valence))),
                emotional_arousal=max(0.0, min(1.0, result.get("emotional_arousal", current.emotional_arousal))),
                energy_budget=max(0.0, min(100.0, current.energy_budget + result.get("energy_delta", 0))),
                decision_confidence=current.decision_confidence,
                patience_threshold=current.patience_threshold,
                risk_tolerance=current.risk_tolerance,
                social_engagement=current.social_engagement,
            )
            return EgoState(
                inner_monologue=result.get("inner_monologue", ""),
                goals=result.get("goals", []),
                social_observations=result.get("social_observations", []),
                updated_cognitive=updated,
            )

        return EgoState(
            inner_monologue="Just going about my day...",
            goals=["get through the day"],
            social_observations=[],
            updated_cognitive=current,
        )
