"""Cognitive and Physical tensors adapted from timepoint-daedalus."""

from pydantic import BaseModel, Field
from typing import Optional


class CognitiveTensor(BaseModel):
    """Cognitive state: knowledge, emotions, energy."""
    knowledge_state: list[str] = Field(default_factory=list)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotional_arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_budget: float = Field(default=100.0, ge=0.0)
    decision_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    patience_threshold: float = Field(default=50.0, ge=0.0)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    social_engagement: float = Field(default=0.8, ge=0.0, le=1.0)


class PhysicalTensor(BaseModel):
    """Physical state: age, health, mobility."""
    age: float
    health_status: float = Field(default=1.0, ge=0.0, le=1.0)
    pain_level: float = Field(default=0.0, ge=0.0, le=1.0)
    pain_location: Optional[str] = None
    mobility: float = Field(default=1.0, ge=0.0, le=1.0)
    stamina: float = Field(default=1.0, ge=0.0, le=1.0)
    sensory_acuity: dict[str, float] = Field(default_factory=dict)
    location: Optional[tuple[float, float]] = None
