"""Event models for simulation events."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class EventType(str, Enum):
    PERSONAL = "personal"
    SOCIAL = "social"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"


class EventImpact(BaseModel):
    """How an event impacts a specific synth."""
    synth_id: str
    emotional_valence_delta: float = 0.0
    emotional_arousal_delta: float = 0.0
    energy_delta: float = 0.0
    knowledge_gained: list[str] = Field(default_factory=list)
    description: str = ""


class SimEvent(BaseModel):
    """A simulation event that affects synths."""
    event_id: str
    tick: int
    event_type: EventType
    title: str
    description: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    affected_synths: list[str] = Field(default_factory=list)
    impacts: list[EventImpact] = Field(default_factory=list)
    source_synth: Optional[str] = None
    ripple_radius: int = 1
