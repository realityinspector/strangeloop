"""Tests for the DAITY simulation engine components."""

import random

import pytest

from strangeloop.knobs.envelope import EnvelopeConfig
from strangeloop.knobs.controller import BehaviorController
from strangeloop.persona2parameters.mapper import PersonaParameterMapper, LLMParams
from strangeloop.schemas.relationship import RelationshipMetrics, RelationshipState
from strangeloop.schemas.synth import (
    BigFive, Backstory, Demographics, Psychographics, SynthProfile,
)
from strangeloop.schemas.tensor import CognitiveTensor
from strangeloop.tessera_ct.assembler import TesseraAssembler


class TestEnvelope:
    def test_attack_phase(self):
        env = EnvelopeConfig(attack=1.0, decay=0.0, sustain=0.8, release=0.0)
        assert env.intensity_at(0.0) == pytest.approx(0.0, abs=0.01)
        assert env.intensity_at(0.125) == pytest.approx(0.5, abs=0.05)
        # At 0.24 (just before boundary), still in attack, near peak
        assert env.intensity_at(0.24) == pytest.approx(0.96, abs=0.05)
        # At 0.25 (boundary), transitions to sustain = 0.8
        assert env.intensity_at(0.25) == pytest.approx(0.8, abs=0.05)

    def test_sustain_phase(self):
        env = EnvelopeConfig(attack=0.0, decay=0.0, sustain=0.6, release=0.0)
        assert env.intensity_at(0.5) == pytest.approx(0.6, abs=0.01)

    def test_release_phase(self):
        env = EnvelopeConfig(attack=0.0, decay=0.0, sustain=0.8, release=1.0)
        val = env.intensity_at(0.99)
        assert val < 0.1  # should be near zero at end

    def test_full_envelope(self):
        env = EnvelopeConfig(attack=0.5, decay=0.5, sustain=0.7, release=0.5)
        # Should never go negative
        for p in [i / 20 for i in range(21)]:
            assert 0.0 <= env.intensity_at(p) <= 1.0

    def test_default_envelope(self):
        env = EnvelopeConfig()
        assert env.intensity_at(0.5) == pytest.approx(0.8, abs=0.01)


class TestPersonaParameterMapper:
    def setup_method(self):
        self.mapper = PersonaParameterMapper()
        self.calm_synth = SynthProfile(
            synth_id="s1", name="Test",
            demographics=Demographics(age=40, gender="male", occupation="accountant"),
            psychographics=Psychographics(
                big_five=BigFive(openness=0.3, conscientiousness=0.9,
                                 extraversion=0.2, agreeableness=0.6, neuroticism=0.1),
                vocabulary_complexity="simple",
                communication_style="terse",
            ),
        )
        self.intense_synth = SynthProfile(
            synth_id="s2", name="Test2",
            demographics=Demographics(age=25, gender="female", occupation="artist"),
            psychographics=Psychographics(
                big_five=BigFive(openness=0.9, conscientiousness=0.3,
                                 extraversion=0.9, agreeableness=0.4, neuroticism=0.8),
                vocabulary_complexity="complex",
                communication_style="verbose",
            ),
        )

    def test_calm_character(self):
        cognitive = CognitiveTensor(emotional_valence=0.0, emotional_arousal=0.1)
        params = self.mapper.map(self.calm_synth, cognitive)
        assert params.temperature < 0.6
        assert params.top_p < 0.85

    def test_intense_character(self):
        cognitive = CognitiveTensor(emotional_valence=0.8, emotional_arousal=0.9)
        params = self.mapper.map(self.intense_synth, cognitive)
        assert params.temperature > 0.7
        assert params.top_p > 0.90

    def test_pattern_affects_tokens(self):
        cognitive = CognitiveTensor()
        banter = self.mapper.map(self.calm_synth, cognitive, "banter")
        confession = self.mapper.map(self.intense_synth, cognitive, "confession")
        assert confession.max_tokens > banter.max_tokens


class TestTesseraAssembler:
    def test_assemble_basic(self):
        assembler = TesseraAssembler()
        synth = SynthProfile(
            synth_id="s1", name="Alice",
            demographics=Demographics(age=30, gender="female", occupation="teacher"),
        )
        cognitive = CognitiveTensor(emotional_valence=0.3, emotional_arousal=0.4)

        prompt = assembler.assemble(
            synth=synth, cognitive=cognitive, events=[],
            tick=3, total_ticks=10,
        )

        assert "Alice" in prompt
        assert "teacher" in prompt
        assert "Tick 3/10" in prompt

    def test_assemble_with_relationship(self):
        assembler = TesseraAssembler()
        synth = SynthProfile(
            synth_id="s1", name="Alice",
            demographics=Demographics(age=30, gender="female", occupation="teacher"),
        )
        cognitive = CognitiveTensor()
        rel = RelationshipState(
            synth_a="s1", synth_b="s2",
            description="Old friends who recently had a disagreement",
            relationship_type="friend",
            metrics=RelationshipMetrics(trust_level=0.6, emotional_bond=0.4),
        )

        prompt = assembler.assemble(
            synth=synth, cognitive=cognitive, events=[],
            tick=1, total_ticks=10,
            relationship=rel, partner_name="Bob",
        )

        assert "Bob" in prompt
        assert "disagreement" in prompt


class TestBehaviorController:
    def test_select_pattern(self):
        envelope = EnvelopeConfig(sustain=0.6)
        ctrl = BehaviorController(envelope, ["BANTER", "DEBATE", "ARGUMENT"])
        pattern = ctrl.select_pattern(5, 10, rng=random.Random(42))
        assert pattern in ["BANTER", "DEBATE", "ARGUMENT"]

    def test_high_intensity_favors_argument(self):
        envelope = EnvelopeConfig(attack=0.0, decay=0.0, sustain=1.0, release=0.0)
        ctrl = BehaviorController(envelope, ["BANTER", "ARGUMENT"])
        counts = {"BANTER": 0, "ARGUMENT": 0}
        rng = random.Random(42)
        for _ in range(100):
            p = ctrl.select_pattern(5, 10, rng=rng)
            counts[p] += 1
        # At high intensity, ARGUMENT should be selected more often
        assert counts["ARGUMENT"] > counts["BANTER"]
