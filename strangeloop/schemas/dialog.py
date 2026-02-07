"""Dialog models adapted from timepoint-daedalus."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DialogTurn(BaseModel):
    """Single turn in a conversation."""
    speaker: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotional_tone: Optional[str] = None
    knowledge_references: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    inner_monologue: Optional[str] = None
    llm_params: Optional[dict] = None


class DialogData(BaseModel):
    """A complete conversation between synths."""
    dialog_id: str
    tick: int
    participants: list[str]
    turns: list[DialogTurn] = Field(default_factory=list)
    conversation_pattern: str = "GENERAL"
    information_exchanged: list[str] = Field(default_factory=list)
    relationship_impacts: dict[str, float] = Field(default_factory=dict)
