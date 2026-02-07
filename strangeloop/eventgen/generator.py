"""Event generation and injection into the simulation.

Events are the external stimuli that keep conversations interesting.
Probability scales with tick number and relationship tension.
"""

import random
import uuid

import networkx as nx
from rich.console import Console

from strangeloop.gpt_genius.client import LLMClient
from strangeloop.schemas.event import EventImpact, EventType, SimEvent
from strangeloop.schemas.relationship import RelationshipState
from strangeloop.schemas.synth import SynthProfile

console = Console()

EVENT_SYSTEM = """You are an event generator for a social simulation. Given the setting,
characters, and current tensions, generate a plausible event as JSON:
{
  "title": "<short event title>",
  "description": "<1-2 sentence description>",
  "event_type": "<personal|social|economic|environmental>",
  "severity": <0.1-1.0>,
  "affected_synth_ids": ["<synth_id_1>", ...],
  "impacts": [
    {
      "synth_id": "<synth_id>",
      "emotional_valence_delta": <-0.5 to 0.5>,
      "emotional_arousal_delta": <-0.3 to 0.5>,
      "energy_delta": <-20 to 10>,
      "description": "<how this affects them>"
    }
  ]
}

Events should create conversational opportunities. Good events:
- Create information asymmetry (some know, some don't)
- Generate emotional reactions that differ by personality
- Force characters to make decisions or take sides
- Escalate or resolve existing tensions"""


class EventGenerator:
    """Generate events that inject drama into the simulation."""

    def __init__(
        self,
        client: LLMClient,
        event_frequency: float = 0.3,
        allowed_types: list[str] | None = None,
    ):
        self.client = client
        self.event_frequency = event_frequency
        self.allowed_types = allowed_types or ["personal", "social"]

    def maybe_generate(
        self,
        tick: int,
        total_ticks: int,
        graph: nx.Graph,
        synths: dict[str, SynthProfile],
        relationships: dict[tuple[str, str], RelationshipState],
        context: str,
        rng: random.Random,
    ) -> list[SimEvent]:
        """Possibly generate events for this tick. Returns empty list if none triggered."""
        # Base probability increases with tick number (more happens over time)
        progress = tick / max(total_ticks, 1)
        prob = self.event_frequency * (0.5 + progress * 0.5)

        # Tension in relationships increases event probability
        tensions = [
            abs(r.metrics.emotional_bond) + (1 - r.metrics.trust_level)
            for r in relationships.values()
        ]
        if tensions:
            avg_tension = sum(tensions) / len(tensions)
            prob += avg_tension * 0.15

        if rng.random() > prob:
            return []

        return [self._generate_event(tick, graph, synths, relationships, context)]

    def _generate_event(
        self,
        tick: int,
        graph: nx.Graph,
        synths: dict[str, SynthProfile],
        relationships: dict[tuple[str, str], RelationshipState],
        context: str,
    ) -> SimEvent:
        """Generate a single event via LLM."""
        synth_summaries = "\n".join(
            f"- {s.name} ({sid}): {s.demographics.occupation}, mood={s.psychographics.emotional_baseline}"
            for sid, s in list(synths.items())[:10]
        )

        # Find most tense relationship for event seeding
        tense_rel = ""
        if relationships:
            most_tense = min(relationships.values(), key=lambda r: r.metrics.trust_level)
            tense_rel = f"Tension between {most_tense.synth_a} and {most_tense.synth_b}: {most_tense.description}"

        user_prompt = (
            f"Setting: {context}\n"
            f"Current tick: {tick}\n"
            f"Characters:\n{synth_summaries}\n"
            f"Allowed event types: {', '.join(self.allowed_types)}\n"
            f"Current tension: {tense_rel}\n"
            f"Generate an event."
        )

        result = self.client.call(
            messages=[
                {"role": "system", "content": EVENT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=400,
            json_mode=True,
            fallback_response={
                "title": "An unexpected development",
                "description": "Something happens that gets people talking.",
                "event_type": "social",
                "severity": 0.4,
                "affected_synth_ids": list(synths.keys())[:2],
                "impacts": [],
            },
        )

        return self._parse_event(result, tick)

    def _parse_event(self, result: dict, tick: int) -> SimEvent:
        """Parse LLM result into SimEvent."""
        if isinstance(result, dict) and "error" not in result:
            event_type_str = result.get("event_type", "social")
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                event_type = EventType.SOCIAL

            impacts = []
            for imp in result.get("impacts", []):
                if isinstance(imp, dict):
                    impacts.append(EventImpact(
                        synth_id=imp.get("synth_id", "unknown"),
                        emotional_valence_delta=imp.get("emotional_valence_delta", 0.0),
                        emotional_arousal_delta=imp.get("emotional_arousal_delta", 0.0),
                        energy_delta=imp.get("energy_delta", 0.0),
                        description=imp.get("description", ""),
                    ))

            return SimEvent(
                event_id=f"evt_{tick}_{uuid.uuid4().hex[:6]}",
                tick=tick,
                event_type=event_type,
                title=result.get("title", "Unknown event"),
                description=result.get("description", ""),
                severity=max(0.0, min(1.0, result.get("severity", 0.5))),
                affected_synths=result.get("affected_synth_ids", []),
                impacts=impacts,
            )

        return SimEvent(
            event_id=f"evt_{tick}_{uuid.uuid4().hex[:6]}",
            tick=tick,
            event_type=EventType.SOCIAL,
            title="An unexpected development",
            description="Something happens that gets people talking.",
            severity=0.4,
        )
