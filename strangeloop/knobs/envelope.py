"""ADSR Envelope for simulation intensity modulation.

Adapted from timepoint-daedalus/synth/envelope.py.

Controls how simulation intensity evolves:
- Attack: onset of conversation intensity
- Decay: settling after initial peak
- Sustain: steady state intensity
- Release: winding down
"""

from pydantic import BaseModel, Field


class EnvelopeConfig(BaseModel):
    """ADSR envelope controlling simulation intensity over its lifetime."""
    attack: float = Field(default=0.1, ge=0.0, le=1.0)
    decay: float = Field(default=0.2, ge=0.0, le=1.0)
    sustain: float = Field(default=0.8, ge=0.0, le=1.0)
    release: float = Field(default=0.3, ge=0.0, le=1.0)

    def intensity_at(self, progress: float, total_ticks: int = 1) -> float:
        """Calculate intensity multiplier at a given progress point (0.0-1.0).

        The envelope divides the simulation into four phases:
        - Attack (first 0-25%): ramp up from 0 to 1
        - Decay (next 0-25%): drop from 1 to sustain level
        - Sustain (middle): hold at sustain level
        - Release (last 0-25%): fade from sustain to 0

        TODO: This is a design choice point. The current implementation
        uses linear interpolation within each phase. You might prefer:
        - Exponential curves for more natural attack/release
        - S-curves (sigmoid) for smoother transitions
        - Custom easing functions per phase
        """
        progress = max(0.0, min(1.0, progress))

        a_end = self.attack * 0.25
        d_end = a_end + self.decay * 0.25
        r_start = 1.0 - self.release * 0.25

        if progress < a_end and a_end > 0:
            return progress / a_end
        elif progress < d_end and (d_end - a_end) > 0:
            decay_progress = (progress - a_end) / (d_end - a_end)
            return 1.0 - (1.0 - self.sustain) * decay_progress
        elif progress < r_start:
            return self.sustain
        elif (1.0 - r_start) > 0:
            release_progress = (progress - r_start) / (1.0 - r_start)
            return self.sustain * (1.0 - release_progress)
        else:
            return self.sustain

    def __repr__(self) -> str:
        return f"EnvelopeConfig(A={self.attack:.2f}, D={self.decay:.2f}, S={self.sustain:.2f}, R={self.release:.2f})"


DEFAULT_ENVELOPE = EnvelopeConfig()
