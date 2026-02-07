"""SynthProfile and related models for synthetic personas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DetailLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"


class Demographics(BaseModel):
    """Basic demographic information for a synth."""
    age: int = Field(ge=0, le=120)
    gender: str
    occupation: str
    education_level: str = ""
    location: str = ""
    ethnicity: str = ""
    socioeconomic_status: str = ""


class BigFive(BaseModel):
    """Big Five personality traits, each 0.0-1.0."""
    openness: float = Field(default=0.5, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0)


class Psychographics(BaseModel):
    """Personality and behavioral profile."""
    big_five: BigFive = Field(default_factory=BigFive)
    values: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    communication_style: str = "conversational"
    vocabulary_complexity: str = "standard"  # simple, standard, complex
    emotional_baseline: str = "neutral"  # calm, anxious, cheerful, melancholic


class Backstory(BaseModel):
    """Character backstory and motivations."""
    summary: str = ""
    key_events: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    relationships_described: list[str] = Field(default_factory=list)


class SocialMediaBehavior(BaseModel):
    """How the synth behaves in social contexts."""
    posting_frequency: str = "moderate"  # rare, moderate, frequent
    topics_of_interest: list[str] = Field(default_factory=list)
    conflict_style: str = "avoidant"  # avoidant, competitive, collaborative, accommodating
    influence_seeking: float = Field(default=0.5, ge=0.0, le=1.0)
    group_conformity: float = Field(default=0.5, ge=0.0, le=1.0)


class SynthProfile(BaseModel):
    """Complete profile for a synthetic persona (synth)."""
    synth_id: str
    name: str
    demographics: Demographics
    psychographics: Psychographics = Field(default_factory=Psychographics)
    backstory: Backstory = Field(default_factory=Backstory)
    social_behavior: SocialMediaBehavior = Field(default_factory=SocialMediaBehavior)
    system_prompt: str = ""
    voice_description: str = ""
    eigenvector_centrality: float = 0.0
    node_id: Optional[str] = None
