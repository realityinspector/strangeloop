"""Map personality traits + emotional state to LLM sampling parameters.

Adapted from persona2params director pattern. The core idea: a synth's
personality and current emotional state should directly control how the
LLM generates their speech.

Temperature ranges (from prompts.json):
  0.3-0.5: calm, measured, formal, logical
  0.6-0.8: normal conversation, moderate emotion
  0.9-1.3: high emotion (anger, passion, joy), impulsive

Top_p ranges:
  0.70-0.80: simple, direct, everyday language
  0.85-0.90: standard conversational range
  0.92-0.98: rich vocabulary, nuanced, complex

Max_tokens by conversation pattern position:
  Brief reactions: 15-40
  Standard exchanges: 50-100
  Longer responses: 100-200
  Monologues: 150-300
"""

from dataclasses import dataclass

from strangeloop.schemas.synth import SynthProfile
from strangeloop.schemas.tensor import CognitiveTensor


@dataclass
class LLMParams:
    """LLM sampling parameters derived from personality."""
    temperature: float
    top_p: float
    max_tokens: int


# Pattern position -> base token range
PATTERN_TOKENS: dict[str, tuple[int, int]] = {
    "question": (15, 40),
    "answer": (40, 120),
    "debate_turn": (50, 100),
    "banter": (15, 40),
    "argument_claim": (60, 100),
    "argument_rebuttal": (80, 120),
    "confession": (120, 250),
    "storytelling": (150, 250),
    "brief_reaction": (15, 40),
    "standard": (50, 120),
}


class PersonaParameterMapper:
    """Maps synth personality + emotional state to LLM parameters."""

    def map(
        self,
        synth: SynthProfile,
        cognitive: CognitiveTensor,
        pattern_position: str = "standard",
        intensity_multiplier: float = 1.0,
    ) -> LLMParams:
        """Derive LLM params from personality and emotional state.

        This is where personality becomes behavior: a neurotic character
        with high arousal gets high temperature (unpredictable), while a
        conscientious, calm character gets low temperature (precise).
        """
        temperature = self._compute_temperature(synth, cognitive, intensity_multiplier)
        top_p = self._compute_top_p(synth)
        max_tokens = self._compute_max_tokens(synth, pattern_position, intensity_multiplier)

        return LLMParams(
            temperature=round(temperature, 2),
            top_p=round(top_p, 2),
            max_tokens=max_tokens,
        )

    def _compute_temperature(
        self,
        synth: SynthProfile,
        cognitive: CognitiveTensor,
        intensity: float,
    ) -> float:
        """Temperature from emotional arousal + neuroticism + extraversion.

        High arousal + high neuroticism = hot (impulsive, emotional)
        Low arousal + low neuroticism = cool (measured, logical)
        """
        bf = synth.psychographics.big_five
        arousal = cognitive.emotional_arousal
        valence_extremity = abs(cognitive.emotional_valence)

        # Base: 0.5 (neutral) shifted by arousal
        base = 0.5 + arousal * 0.4

        # Neuroticism amplifies emotional temperature
        base += (bf.neuroticism - 0.5) * 0.3

        # Extraversion adds expressiveness
        base += (bf.extraversion - 0.5) * 0.15

        # Extreme emotions (positive or negative) push temperature up
        base += valence_extremity * 0.2

        # Apply ADSR intensity multiplier
        base *= intensity

        return max(0.3, min(1.3, base))

    def _compute_top_p(self, synth: SynthProfile) -> float:
        """Top_p from vocabulary complexity + openness.

        Complex vocabulary + high openness = wider sampling
        Simple vocabulary + low openness = narrower, predictable
        """
        bf = synth.psychographics.big_five
        vocab = synth.psychographics.vocabulary_complexity

        vocab_base = {"simple": 0.75, "standard": 0.87, "complex": 0.95}.get(vocab, 0.87)

        # Openness widens vocabulary range
        openness_shift = (bf.openness - 0.5) * 0.08

        return max(0.70, min(0.98, vocab_base + openness_shift))

    def _compute_max_tokens(
        self,
        synth: SynthProfile,
        pattern_position: str,
        intensity: float,
    ) -> int:
        """Max tokens from pattern position + communication style + extraversion."""
        token_range = PATTERN_TOKENS.get(pattern_position, PATTERN_TOKENS["standard"])
        base_min, base_max = token_range

        # Communication style shifts the range
        style = synth.psychographics.communication_style
        style_mult = {
            "terse": 0.6, "casual": 0.85, "conversational": 1.0,
            "verbose": 1.3, "formal": 1.1, "academic": 1.2,
        }.get(style, 1.0)

        # Extraversion affects verbosity
        ext = synth.psychographics.big_five.extraversion
        ext_mult = 0.8 + ext * 0.4  # 0.8x to 1.2x

        midpoint = (base_min + base_max) / 2
        tokens = midpoint * style_mult * ext_mult * intensity

        return max(base_min, min(base_max * 2, int(tokens)))
