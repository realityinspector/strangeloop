"""Relationship description and metrics generation.

Given two SynthProfiles, produces RelationshipMetrics and descriptive state.
"""

import networkx as nx
from rich.console import Console

from strangeloop.gpt_genius.client import LLMClient
from strangeloop.schemas.relationship import RelationshipMetrics, RelationshipState
from strangeloop.schemas.synth import SynthProfile

console = Console()

RELATIONSHIP_SYSTEM = """You are a social dynamics analyst. Given two character profiles,
generate a realistic relationship between them as JSON:
{
  "description": "<1-2 sentence description of their relationship>",
  "relationship_type": "<friend|rival|mentor|colleague|neighbor|family|romantic|adversary|acquaintance>",
  "trust_level": <0.0-1.0>,
  "emotional_bond": <-1.0 to 1.0>,
  "power_dynamic": <-1.0 to 1.0, negative = A is subordinate>,
  "belief_alignment": <-1.0 to 1.0>,
  "shared_knowledge": <0-10, number of shared knowledge items>
}

Make relationships nuanced - not everyone is friends. Consider:
- Age gaps, occupational overlap, personality compatibility
- Power differentials from status, expertise, or personality
- Potential sources of conflict or mutual benefit"""

RELATIONSHIP_UPDATE_SYSTEM = """You are a social dynamics analyst. Given a conversation between
two characters and their current relationship, output updated metrics as JSON:
{
  "trust_level": <0.0-1.0>,
  "emotional_bond": <-1.0 to 1.0>,
  "power_dynamic": <-1.0 to 1.0>,
  "belief_alignment": <-1.0 to 1.0>,
  "interaction_count_delta": 1,
  "recent_event": "<brief description of what changed>"
}

Small changes are realistic. Trust builds slowly, breaks quickly.
Emotional bonds shift with shared experiences. Power dynamics
shift when someone demonstrates competence or vulnerability."""


class RelationshipGenerator:
    """Generate and update relationships between synths."""

    def __init__(self, client: LLMClient):
        self.client = client

    def generate_all(
        self,
        graph: nx.Graph,
        synths: dict[str, SynthProfile],
    ) -> dict[tuple[str, str], RelationshipState]:
        """Generate relationship states for all edges in the graph."""
        relationships: dict[tuple[str, str], RelationshipState] = {}

        edges = list(graph.edges())
        for i, (a, b) in enumerate(edges):
            console.print(f"  [dim]Generating relationship {i+1}/{len(edges)}: {a} <-> {b}[/dim]")

            synth_a = synths.get(a)
            synth_b = synths.get(b)
            if not synth_a or not synth_b:
                relationships[(a, b)] = self._fallback_relationship(a, b)
                continue

            user_prompt = (
                f"Character A: {synth_a.name}\n"
                f"  Age: {synth_a.demographics.age}, Occupation: {synth_a.demographics.occupation}\n"
                f"  Personality: O={synth_a.psychographics.big_five.openness:.1f} "
                f"C={synth_a.psychographics.big_five.conscientiousness:.1f} "
                f"E={synth_a.psychographics.big_five.extraversion:.1f} "
                f"A={synth_a.psychographics.big_five.agreeableness:.1f} "
                f"N={synth_a.psychographics.big_five.neuroticism:.1f}\n"
                f"  Backstory: {synth_a.backstory.summary[:200]}\n\n"
                f"Character B: {synth_b.name}\n"
                f"  Age: {synth_b.demographics.age}, Occupation: {synth_b.demographics.occupation}\n"
                f"  Personality: O={synth_b.psychographics.big_five.openness:.1f} "
                f"C={synth_b.psychographics.big_five.conscientiousness:.1f} "
                f"E={synth_b.psychographics.big_five.extraversion:.1f} "
                f"A={synth_b.psychographics.big_five.agreeableness:.1f} "
                f"N={synth_b.psychographics.big_five.neuroticism:.1f}\n"
                f"  Backstory: {synth_b.backstory.summary[:200]}"
            )

            result = self.client.call(
                messages=[
                    {"role": "system", "content": RELATIONSHIP_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
                json_mode=True,
            )

            relationships[(a, b)] = self._parse_relationship(result, a, b)

        return relationships

    def update_relationship(
        self,
        state: RelationshipState,
        synth_a: SynthProfile,
        synth_b: SynthProfile,
        conversation_summary: str,
    ) -> RelationshipState:
        """Update a relationship based on a conversation."""
        user_prompt = (
            f"Current relationship: {state.description}\n"
            f"Trust: {state.metrics.trust_level:.2f}, Bond: {state.metrics.emotional_bond:.2f}\n"
            f"Power: {state.metrics.power_dynamic:.2f}, Alignment: {state.metrics.belief_alignment:.2f}\n\n"
            f"Conversation:\n{conversation_summary}"
        )

        result = self.client.call(
            messages=[
                {"role": "system", "content": RELATIONSHIP_UPDATE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=200,
            json_mode=True,
        )

        if isinstance(result, dict) and "error" not in result:
            state.metrics.trust_level = max(0.0, min(1.0, result.get("trust_level", state.metrics.trust_level)))
            state.metrics.emotional_bond = max(-1.0, min(1.0, result.get("emotional_bond", state.metrics.emotional_bond)))
            state.metrics.power_dynamic = max(-1.0, min(1.0, result.get("power_dynamic", state.metrics.power_dynamic)))
            state.metrics.belief_alignment = max(-1.0, min(1.0, result.get("belief_alignment", state.metrics.belief_alignment)))
            state.metrics.interaction_count += result.get("interaction_count_delta", 1)
            if "recent_event" in result:
                state.recent_events.append(result["recent_event"])
                state.recent_events = state.recent_events[-5:]  # keep last 5

        return state

    def _parse_relationship(self, result: dict, a: str, b: str) -> RelationshipState:
        """Parse LLM result into RelationshipState."""
        if isinstance(result, dict) and "error" not in result:
            return RelationshipState(
                synth_a=a,
                synth_b=b,
                description=result.get("description", ""),
                relationship_type=result.get("relationship_type", "acquaintance"),
                metrics=RelationshipMetrics(
                    trust_level=max(0.0, min(1.0, result.get("trust_level", 0.5))),
                    emotional_bond=max(-1.0, min(1.0, result.get("emotional_bond", 0.0))),
                    power_dynamic=max(-1.0, min(1.0, result.get("power_dynamic", 0.0))),
                    belief_alignment=max(-1.0, min(1.0, result.get("belief_alignment", 0.0))),
                    shared_knowledge=result.get("shared_knowledge", 0),
                ),
            )
        return self._fallback_relationship(a, b)

    def _fallback_relationship(self, a: str, b: str) -> RelationshipState:
        """Create minimal fallback relationship."""
        return RelationshipState(
            synth_a=a,
            synth_b=b,
            description="Acquaintances with limited interaction.",
            relationship_type="acquaintance",
        )
