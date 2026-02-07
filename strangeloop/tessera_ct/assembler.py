"""Build the 4-6 part Conversational Tensor (system prompt).

The Conversational Tensor assembles context from multiple sources
into a unified system prompt for LLM calls:

1. Persona details (who you are)
2. Inner state / goals (what you're thinking)
3. Active events (what's happening around you)
4. Time context (when this is happening)
5. Setting description (optional: where you are)
6. Social context (optional: relationship with conversation partner)
"""

from strangeloop.schemas.synth import SynthProfile
from strangeloop.schemas.tensor import CognitiveTensor
from strangeloop.schemas.relationship import RelationshipState
from strangeloop.schemas.event import SimEvent

# Roleplayer suffix adapted from persona2params/prompts.json
ROLEPLAYER_SUFFIX = (
    "Embody your character with complete consistency. Maintain your distinctive voice, "
    "speech patterns, vocabulary level, and mannerisms throughout. Let your character's "
    "background, culture, motivations, fears, and biases shape every response. Show emotional "
    "range appropriate to the situation. Include subtext and what your character doesn't say. "
    "React authentically to the stakes that matter to YOU as this character.\n\n"
    "Respond STRICTLY with ONLY the spoken dialog from your character's perspective. "
    "NO character names, NO prefixes, NO actions in asterisks, NO descriptions. "
    "Pure dialog only."
)


class TesseraAssembler:
    """Assembles the multi-part system prompt for synth LLM calls."""

    def assemble(
        self,
        synth: SynthProfile,
        cognitive: CognitiveTensor,
        events: list[SimEvent],
        tick: int,
        total_ticks: int,
        time_label: str = "",
        setting: str = "",
        relationship: RelationshipState | None = None,
        partner_name: str = "",
        inner_monologue: str = "",
        goals: list[str] | None = None,
    ) -> str:
        """Build the full conversational tensor system prompt."""
        parts: list[str] = []

        # Part 1: Persona
        parts.append(self._persona_part(synth))

        # Part 2: Inner state and goals
        parts.append(self._inner_state_part(cognitive, inner_monologue, goals))

        # Part 3: Active events
        if events:
            parts.append(self._events_part(events))

        # Part 4: Time context
        parts.append(self._time_part(tick, total_ticks, time_label))

        # Part 5: Setting (optional)
        if setting:
            parts.append(f"SETTING: {setting}")

        # Part 6: Social context (optional)
        if relationship and partner_name:
            parts.append(self._social_part(relationship, partner_name))

        parts.append(ROLEPLAYER_SUFFIX)

        return "\n\n".join(parts)

    def _persona_part(self, synth: SynthProfile) -> str:
        bf = synth.psychographics.big_five
        return (
            f"You are {synth.name}, a {synth.demographics.age}-year-old "
            f"{synth.demographics.occupation}.\n"
            f"Voice: {synth.voice_description or synth.psychographics.communication_style}\n"
            f"Personality: Openness={bf.openness:.1f}, Conscientiousness={bf.conscientiousness:.1f}, "
            f"Extraversion={bf.extraversion:.1f}, Agreeableness={bf.agreeableness:.1f}, "
            f"Neuroticism={bf.neuroticism:.1f}\n"
            f"Values: {', '.join(synth.psychographics.values[:5]) or 'not specified'}\n"
            f"Background: {synth.backstory.summary or 'not specified'}"
        )

    def _inner_state_part(
        self,
        cognitive: CognitiveTensor,
        inner_monologue: str,
        goals: list[str] | None,
    ) -> str:
        mood = "positive" if cognitive.emotional_valence > 0.2 else "negative" if cognitive.emotional_valence < -0.2 else "neutral"
        energy = "high" if cognitive.energy_budget > 70 else "low" if cognitive.energy_budget < 30 else "moderate"

        lines = [
            f"INNER STATE: Mood is {mood} (valence: {cognitive.emotional_valence:.1f}), "
            f"arousal: {cognitive.emotional_arousal:.1f}, energy: {energy}.",
        ]
        if inner_monologue:
            lines.append(f"You're thinking: {inner_monologue}")
        if goals:
            lines.append(f"Current goals: {'; '.join(goals[:3])}")
        return "\n".join(lines)

    def _events_part(self, events: list[SimEvent]) -> str:
        lines = ["RECENT EVENTS affecting you:"]
        for ev in events[:3]:
            lines.append(f"- [{ev.event_type.value}] {ev.title}: {ev.description[:100]}")
        return "\n".join(lines)

    def _time_part(self, tick: int, total_ticks: int, time_label: str) -> str:
        progress = tick / max(total_ticks, 1)
        phase = "early" if progress < 0.3 else "middle" if progress < 0.7 else "late"
        label = f" ({time_label})" if time_label else ""
        return f"TIME: Tick {tick}/{total_ticks}{label} [{phase} phase]"

    def _social_part(self, rel: RelationshipState, partner_name: str) -> str:
        return (
            f"RELATIONSHIP with {partner_name}: {rel.description}\n"
            f"Type: {rel.relationship_type}, Trust: {rel.metrics.trust_level:.1f}, "
            f"Bond: {rel.metrics.emotional_bond:.1f}, Power: {rel.metrics.power_dynamic:.1f}"
        )
