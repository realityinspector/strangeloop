"""Built-in scenario presets for quick simulation setup."""

from strangeloop.schemas.simulation import (
    EnvelopeConfig, GraphConfig, SimulationConfig, Topology,
)


def _small_town() -> SimulationConfig:
    return SimulationConfig(
        name="small_town",
        description="A small rural town reacting to a proposed highway bypass",
        context_description=(
            "Millfield is a quiet rural town of 2,000 people. The state DOT has proposed "
            "building a highway bypass that would cut through farmland on the east side. "
            "Residents are divided: some see economic opportunity, others fear losing their "
            "way of life. Town council meeting is next week."
        ),
        graph=GraphConfig(
            num_synths=8,
            topology=Topology.COMMUNITY,
            num_communities=3,
            avg_connections=3,
            edge_density=0.35,
        ),
        num_ticks=5,
        token_budget=80000,
        cost_limit_usd=3.0,
        envelope=EnvelopeConfig(attack=0.3, decay=0.1, sustain=0.8, release=0.2),
        conversation_patterns=["DEBATE", "ARGUMENT", "BANTER", "CONFESSION"],
        event_frequency=0.4,
        event_types=["personal", "social", "economic"],
    )


def _startup() -> SimulationConfig:
    return SimulationConfig(
        name="startup",
        description="A tech startup navigating a pivotal funding round",
        context_description=(
            "NeuralEdge is a 12-person AI startup that just got a term sheet from a major VC. "
            "The founders disagree on whether to take the money (with board seats and control) "
            "or bootstrap with a smaller angel round. The team is stressed, working 14-hour days, "
            "and rumors of layoffs at competitors are making everyone anxious."
        ),
        graph=GraphConfig(
            num_synths=6,
            topology=Topology.SCALE_FREE,
            avg_connections=3,
            edge_density=0.4,
        ),
        num_ticks=5,
        token_budget=60000,
        cost_limit_usd=2.5,
        envelope=EnvelopeConfig(attack=0.2, decay=0.15, sustain=0.9, release=0.15),
        conversation_patterns=["DEBATE", "NEGOTIATION", "ARGUMENT", "BANTER"],
        event_frequency=0.5,
        event_types=["personal", "social", "economic"],
    )


PRESETS: dict[str, SimulationConfig] = {
    "small_town": _small_town(),
    "startup": _startup(),
}


def get_preset(name: str) -> SimulationConfig | None:
    """Get a preset by name, or None if not found."""
    return PRESETS.get(name)
