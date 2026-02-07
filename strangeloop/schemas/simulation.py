"""Simulation configuration and state models."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Topology(str, Enum):
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    COMMUNITY = "community"
    RANDOM = "random"


class GraphConfig(BaseModel):
    """Configuration for the social graph."""
    num_synths: int = Field(default=10, ge=2, le=100)
    topology: Topology = Topology.SMALL_WORLD
    edge_density: float = Field(default=0.3, ge=0.0, le=1.0)
    avg_connections: int = Field(default=4, ge=1)
    rewire_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    num_communities: int = Field(default=3, ge=1)
    seed: Optional[int] = 42


class EnvelopeConfig(BaseModel):
    """ADSR envelope parameters for simulation intensity."""
    attack: float = Field(default=0.2, ge=0.0, le=1.0)
    decay: float = Field(default=0.1, ge=0.0, le=1.0)
    sustain: float = Field(default=0.8, ge=0.0, le=1.0)
    release: float = Field(default=0.3, ge=0.0, le=1.0)


class SimulationConfig(BaseModel):
    """Complete simulation configuration loaded from JSON."""
    name: str = "unnamed_simulation"
    description: str = ""
    seed: Optional[int] = 42
    graph: GraphConfig = Field(default_factory=GraphConfig)
    num_ticks: int = Field(default=10, ge=1)
    time_scale: str = "days"
    start_date: str = "2025-06-01"
    token_budget: int = 100000
    cost_limit_usd: float = 5.0
    default_model: str = "openai/gpt-4o"
    fallback_model: str = "meta-llama/llama-3.1-70b-instruct:free"
    synth_detail_level: str = "standard"
    context_description: str = ""
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    conversation_patterns: list[str] = Field(
        default_factory=lambda: ["DEBATE", "ARGUMENT", "BANTER"]
    )
    event_frequency: float = Field(default=0.3, ge=0.0, le=1.0)
    event_types: list[str] = Field(
        default_factory=lambda: ["personal", "social"]
    )
    animism_mode: bool = False
    output_dir: str = "output"
    verbose: bool = False


class TokenUsage(BaseModel):
    """Tracks token consumption."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost_per_1k: float = 0.005):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.estimated_cost_usd += (prompt + completion) / 1000 * cost_per_1k


class SimulationState(BaseModel):
    """Runtime state of a simulation."""
    config: SimulationConfig
    current_tick: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    is_running: bool = False
    synth_ids: list[str] = Field(default_factory=list)
    active_events: list[str] = Field(default_factory=list)

    def budget_remaining(self) -> int:
        return self.config.token_budget - self.token_usage.total_tokens

    def cost_remaining(self) -> float:
        return self.config.cost_limit_usd - self.token_usage.estimated_cost_usd

    def within_budget(self) -> bool:
        return self.budget_remaining() > 0 and self.cost_remaining() > 0
