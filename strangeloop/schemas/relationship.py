"""Relationship models adapted from timepoint-daedalus."""

from pydantic import BaseModel, Field


class RelationshipMetrics(BaseModel):
    """Quantifies the relationship between two synths."""
    shared_knowledge: int = 0
    belief_alignment: float = Field(default=0.0, ge=-1.0, le=1.0)
    interaction_count: int = 0
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_bond: float = Field(default=0.0, ge=-1.0, le=1.0)
    power_dynamic: float = Field(default=0.0, ge=-1.0, le=1.0)


class RelationshipState(BaseModel):
    """Full relationship state between two synths at a point in time."""
    synth_a: str
    synth_b: str
    description: str = ""
    relationship_type: str = "acquaintance"
    metrics: RelationshipMetrics = Field(default_factory=RelationshipMetrics)
    recent_events: list[str] = Field(default_factory=list)
    tick: int = 0
