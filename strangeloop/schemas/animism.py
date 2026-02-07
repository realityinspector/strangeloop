"""Animistic entity types adapted from timepoint-daedalus."""

from pydantic import BaseModel, Field


class AnimalEntity(BaseModel):
    """Biological state and capabilities of animal entities."""
    species: str
    biological_state: dict[str, float] = Field(default_factory=dict)
    training_level: float = Field(default=0.0, ge=0.0, le=1.0)
    goals: list[str] = Field(default_factory=list)
    sensory_capabilities: dict[str, float] = Field(default_factory=dict)
    physical_capabilities: dict[str, float] = Field(default_factory=dict)


class BuildingEntity(BaseModel):
    """Physical and functional properties of building entities."""
    structural_integrity: float = Field(default=1.0, ge=0.0, le=1.0)
    capacity: int = 0
    age: int = 0
    maintenance_state: float = Field(default=1.0, ge=0.0, le=1.0)
    constraints: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)


class ObjectEntity(BaseModel):
    """Properties of inanimate objects given animistic presence."""
    object_type: str = ""
    condition: float = Field(default=1.0, ge=0.0, le=1.0)
    symbolic_meaning: str = ""
    affordances: list[str] = Field(default_factory=list)
    owner: str = ""
