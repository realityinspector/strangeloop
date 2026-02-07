"""Integration tests with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strangeloop.schemas.simulation import SimulationConfig


MOCK_SYNTH = {
    "name": "Test Person",
    "demographics": {"age": 35, "gender": "male", "occupation": "engineer"},
    "psychographics": {
        "big_five": {"openness": 0.5, "conscientiousness": 0.5,
                     "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
        "values": ["truth"], "interests": ["coding"],
        "communication_style": "conversational",
        "vocabulary_complexity": "standard",
        "emotional_baseline": "neutral",
    },
    "backstory": {"summary": "A regular person.", "key_events": [], "motivations": ["work"],
                   "fears": ["failure"], "secrets": []},
    "social_behavior": {"conflict_style": "collaborative", "influence_seeking": 0.5,
                         "group_conformity": 0.5},
    "voice_description": "Plain spoken and direct.",
}

MOCK_REL = {
    "description": "Colleagues who work together.",
    "relationship_type": "colleague",
    "trust_level": 0.6,
    "emotional_bond": 0.2,
    "power_dynamic": 0.0,
    "belief_alignment": 0.3,
    "shared_knowledge": 3,
}

MOCK_EVENT = {
    "title": "Office Announcement",
    "description": "The manager announces a team restructure.",
    "event_type": "social",
    "severity": 0.5,
    "affected_synth_ids": ["synth_0", "synth_1"],
    "impacts": [
        {"synth_id": "synth_0", "emotional_valence_delta": -0.1,
         "emotional_arousal_delta": 0.2, "energy_delta": -5,
         "description": "Worried about position"},
    ],
}

MOCK_EGO = {
    "inner_monologue": "I wonder what this means for me.",
    "goals": ["find out more", "stay calm"],
    "social_observations": ["Bob seems nervous too"],
    "emotional_valence": 0.0,
    "emotional_arousal": 0.4,
    "energy_delta": -3,
}

MOCK_REL_UPDATE = {
    "trust_level": 0.65,
    "emotional_bond": 0.25,
    "power_dynamic": 0.0,
    "belief_alignment": 0.3,
    "interaction_count_delta": 1,
    "recent_event": "Shared concerns about restructure",
}


def _create_mock_client():
    """Create a mock LLM client that returns appropriate responses."""
    client = MagicMock()
    call_count = [0]

    def mock_call(messages, **kwargs):
        call_count[0] += 1
        json_mode = kwargs.get("json_mode", False)

        if json_mode:
            # Determine what kind of call based on system prompt content
            system = messages[0]["content"] if messages else ""
            if "character architect" in system:
                return MOCK_SYNTH
            elif "social dynamics analyst" in system:
                if "conversation" in (messages[-1]["content"] if len(messages) > 1 else ""):
                    return MOCK_REL_UPDATE
                return MOCK_REL
            elif "event generator" in system:
                return MOCK_EVENT
            elif "inner mind" in system:
                return MOCK_EGO
            return MOCK_SYNTH
        else:
            return "Well, that's certainly something to think about."

    client.call.side_effect = mock_call
    client.token_usage = MagicMock()
    client.token_usage.total_tokens = 0
    client.token_usage.estimated_cost_usd = 0.0
    return client


class TestIntegration:
    def test_full_simulation_small(self, tmp_path):
        """Run a minimal simulation with mocked LLM."""
        config = SimulationConfig(
            name="test_sim",
            num_ticks=2,
            graph={"num_synths": 3, "topology": "small_world", "seed": 42},
            token_budget=999999,
            cost_limit_usd=999.0,
            context_description="Test scenario",
            output_dir=str(tmp_path / "output"),
        )

        mock_client = _create_mock_client()

        from strangeloop.daity.engine import SimulationEngine

        engine = SimulationEngine(config)
        engine.client = mock_client
        engine.synth_gen.client = mock_client
        engine.rel_gen.client = mock_client
        engine.event_gen.client = mock_client
        engine.ego_loop.client = mock_client

        state = engine.run()

        assert state.current_tick == 2
        assert len(engine.synths) == 3
        assert len(engine.relationships) > 0
        assert len(engine.all_dialogs) > 0

    def test_config_loading(self):
        """Test that example configs parse correctly."""
        config_dir = Path(__file__).parent.parent / "config" / "examples"
        for config_file in config_dir.glob("*.json"):
            data = json.loads(config_file.read_text())
            config = SimulationConfig(**data)
            assert config.num_ticks > 0
            assert config.graph.num_synths > 0
