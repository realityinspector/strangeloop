"""Tests for synth generation with mocked LLM responses."""

from unittest.mock import MagicMock

import networkx as nx
import pytest

from strangeloop.brancher.generator import generate_graph
from strangeloop.schemas.simulation import GraphConfig, Topology
from strangeloop.schemas.synth import SynthProfile
from strangeloop.synthgen.generator import SynthGenerator


MOCK_SYNTH_RESPONSE = {
    "name": "Alice Chen",
    "demographics": {
        "age": 34,
        "gender": "female",
        "occupation": "teacher",
        "education_level": "masters",
        "location": "Millfield",
        "ethnicity": "Chinese-American",
        "socioeconomic_status": "middle",
    },
    "psychographics": {
        "big_five": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3,
        },
        "values": ["education", "community", "fairness"],
        "interests": ["gardening", "local politics", "reading"],
        "communication_style": "conversational",
        "vocabulary_complexity": "standard",
        "emotional_baseline": "cheerful",
    },
    "backstory": {
        "summary": "Alice moved to Millfield five years ago to teach at the local school.",
        "key_events": ["moved to Millfield", "won teaching award"],
        "motivations": ["help students succeed", "build community"],
        "fears": ["town losing its character", "being an outsider"],
        "secrets": ["considering leaving for a city job"],
    },
    "social_behavior": {
        "conflict_style": "collaborative",
        "influence_seeking": 0.4,
        "group_conformity": 0.6,
    },
    "voice_description": "Warm and encouraging, uses teaching metaphors, occasionally code-switches.",
}


class TestSynthGenerator:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_client.call.return_value = MOCK_SYNTH_RESPONSE
        self.generator = SynthGenerator(self.mock_client, "standard")

    def test_generate_all(self):
        config = GraphConfig(num_synths=3, topology=Topology.SMALL_WORLD, seed=42)
        graph = generate_graph(config)
        synths = self.generator.generate_all(graph, "A small town")

        assert len(synths) == 3
        for sid, profile in synths.items():
            assert isinstance(profile, SynthProfile)
            assert profile.name == "Alice Chen"
            assert profile.demographics.age == 34

    def test_centrality_ordering(self):
        config = GraphConfig(num_synths=5, topology=Topology.SCALE_FREE, seed=42)
        graph = generate_graph(config)

        call_order = []
        def track_calls(*args, **kwargs):
            call_order.append(kwargs.get("messages", args[0] if args else []))
            return MOCK_SYNTH_RESPONSE

        self.mock_client.call.side_effect = track_calls
        self.generator.generate_all(graph, "Test context")

        assert len(call_order) == 5

    def test_fallback_on_error(self):
        self.mock_client.call.return_value = {"error": "API failed"}
        config = GraphConfig(num_synths=2, topology=Topology.SMALL_WORLD, seed=42)
        graph = generate_graph(config)
        synths = self.generator.generate_all(graph, "Test")

        assert len(synths) == 2
        for profile in synths.values():
            assert profile.name.startswith("Synth ")
