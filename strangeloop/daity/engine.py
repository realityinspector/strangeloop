"""DAITY: The main simulation engine.

Orchestrates the full pipeline:
config -> Brancher -> SynthGen -> RelGen -> tick loop:
  EventGen -> egoLoop -> Tessera-CT -> persona2parameters ->
  gptGenius response -> RelGen update -> state persistence
"""

import json
import random
import re
import uuid
from datetime import datetime
from pathlib import Path

import networkx as nx
from rich.console import Console

from strangeloop.brancher.generator import generate_graph, nodes_by_centrality
from strangeloop.ego_loop.loop import EgoLoop, EgoState
from strangeloop.eventgen.generator import EventGenerator
from strangeloop.gpt_genius.client import LLMClient
from strangeloop.knobs.controller import BehaviorController
from strangeloop.knobs.envelope import EnvelopeConfig as KnobsEnvelope
from strangeloop.persona2parameters.mapper import PersonaParameterMapper
from strangeloop.relgen.generator import RelationshipGenerator
from strangeloop.schemas.dialog import DialogData, DialogTurn
from strangeloop.schemas.event import SimEvent
from strangeloop.schemas.relationship import RelationshipState
from strangeloop.schemas.simulation import SimulationConfig, SimulationState
from strangeloop.schemas.synth import SynthProfile
from strangeloop.schemas.tensor import CognitiveTensor
from strangeloop.storage.store import SimulationStore
from strangeloop.synthgen.generator import SynthGenerator
from strangeloop.tessera_ct.assembler import TesseraAssembler

console = Console()

# Character colors for Rich output (from persona2params)
SYNTH_COLORS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "bright_red", "bright_blue", "cyan", "green", "yellow", "magenta",
]


class SimulationEngine:
    """Orchestrates the full Strange Loop simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = SimulationState(config=config)
        self.client = LLMClient(
            model=config.default_model,
            fallback_model=config.fallback_model,
        )
        self.store = SimulationStore(
            db_path=str(Path(config.output_dir) / f"{config.name}.db")
        )
        self.graph: nx.Graph | None = None
        self.synths: dict[str, SynthProfile] = {}
        self.relationships: dict[tuple[str, str], RelationshipState] = {}
        self.cognitive_states: dict[str, CognitiveTensor] = {}
        self.ego_states: dict[str, EgoState] = {}
        self.all_events: list[SimEvent] = []
        self.all_dialogs: list[DialogData] = []
        self.rng = random.Random(config.seed)

        # Initialize subsystems
        self.synth_gen = SynthGenerator(self.client, config.synth_detail_level)
        self.rel_gen = RelationshipGenerator(self.client)
        self.event_gen = EventGenerator(
            self.client, config.event_frequency, config.event_types
        )
        self.ego_loop = EgoLoop(self.client)
        self.tessera = TesseraAssembler()
        self.param_mapper = PersonaParameterMapper()
        self.envelope = KnobsEnvelope(
            attack=config.envelope.attack,
            decay=config.envelope.decay,
            sustain=config.envelope.sustain,
            release=config.envelope.release,
        )
        self.behavior_ctrl = BehaviorController(
            self.envelope, config.conversation_patterns
        )

    def run(self) -> SimulationState:
        """Execute the full simulation."""
        self.state.is_running = True
        console.print(f"\n[bold]Strange Loop: {self.config.name}[/bold]")
        console.print(f"[dim]{self.config.context_description}[/dim]\n")

        # Phase 1: Generate graph
        console.print("[bold cyan]Phase 1: Generating social graph...[/bold cyan]")
        self.graph = generate_graph(self.config.graph)
        self._print_graph_summary()

        # Phase 2: Generate synths
        console.print("\n[bold cyan]Phase 2: Generating synthetic personas...[/bold cyan]")
        self.synths = self.synth_gen.generate_all(
            self.graph, self.config.context_description
        )
        self._init_cognitive_states()
        self._persist_synths()
        self._print_synth_summary()

        # Phase 3: Generate relationships
        console.print("\n[bold cyan]Phase 3: Generating relationships...[/bold cyan]")
        self.relationships = self.rel_gen.generate_all(self.graph, self.synths)
        self._persist_relationships(tick=0)

        # Phase 4: Tick loop
        console.print(f"\n[bold cyan]Phase 4: Running {self.config.num_ticks} ticks...[/bold cyan]\n")
        for tick in range(1, self.config.num_ticks + 1):
            if not self.state.within_budget():
                console.print(f"\n[bold red]Budget exhausted at tick {tick}.[/bold red]")
                break
            self._run_tick(tick)

        self.state.is_running = False
        self.state.token_usage = self.client.token_usage

        console.print(f"\n[bold green]Simulation complete.[/bold green]")
        console.print(
            f"Tokens used: {self.state.token_usage.total_tokens:,} "
            f"(~${self.state.token_usage.estimated_cost_usd:.2f})"
        )

        return self.state

    def _run_tick(self, tick: int):
        """Execute one simulation tick."""
        self.state.current_tick = tick
        progress = tick / self.config.num_ticks
        intensity = self.envelope.intensity_at(progress)

        console.print(f"[bold]--- Tick {tick}/{self.config.num_ticks} (intensity: {intensity:.2f}) ---[/bold]")

        # 1. Event generation
        events = self.event_gen.maybe_generate(
            tick, self.config.num_ticks, self.graph,
            self.synths, self.relationships,
            self.config.context_description, self.rng,
        )
        for event in events:
            console.print(f"  [yellow]Event: {event.title}[/yellow] - {event.description}")
            self.all_events.append(event)
            self._apply_event_impacts(event)
            self.store.save_event(
                self.config.name, event.event_id, tick,
                event.event_type.value, event.title, event.description,
                event.severity, event.affected_synths,
            )

        # 2. Ego loop for each synth
        for synth_id, synth in self.synths.items():
            cognitive = self.cognitive_states[synth_id]
            synth_events = [e for e in events if synth_id in e.affected_synths]

            last_dialog_summary = self._get_recent_dialog_summary(synth_id)
            ego = self.ego_loop.tick(
                synth, cognitive, synth_events, last_dialog_summary, tick
            )
            self.ego_states[synth_id] = ego
            self.cognitive_states[synth_id] = ego.updated_cognitive

        # 3. Select conversation groups and generate dialog
        groups = self._select_conversation_groups(tick)
        for group in groups:
            if not self.state.within_budget():
                break
            dialog = self._run_conversation(tick, group, events, intensity)
            self.all_dialogs.append(dialog)

    def _select_conversation_groups(self, tick: int) -> list[tuple[str, str]]:
        """Select which synths talk this tick via centrality-weighted edge selection.

        This is a key design choice: how synths pair up each tick.
        Current approach: select edges weighted by the sum of endpoint centralities,
        so high-centrality characters appear in more conversations.
        """
        edges = list(self.graph.edges())
        if not edges:
            return []

        weights = []
        for u, v in edges:
            w = (
                self.graph.nodes[u].get("eigenvector", 0.1)
                + self.graph.nodes[v].get("eigenvector", 0.1)
            )
            weights.append(w)

        total = sum(weights)
        probs = [w / total for w in weights]

        # Select 1-3 conversations per tick depending on graph size
        n_convos = min(
            max(1, self.graph.number_of_nodes() // 4),
            len(edges),
            3,
        )

        selected = set()
        groups = []
        for _ in range(n_convos * 3):  # oversample to handle collisions
            if len(groups) >= n_convos:
                break
            idx = self.rng.choices(range(len(edges)), weights=probs, k=1)[0]
            edge = edges[idx]
            if edge[0] not in selected and edge[1] not in selected:
                selected.add(edge[0])
                selected.add(edge[1])
                groups.append(edge)

        return groups

    def _run_conversation(
        self,
        tick: int,
        pair: tuple[str, str],
        events: list[SimEvent],
        intensity: float,
    ) -> DialogData:
        """Run a conversation between two synths."""
        a_id, b_id = pair
        synth_a = self.synths[a_id]
        synth_b = self.synths[b_id]
        rel = self.relationships.get((a_id, b_id)) or self.relationships.get((b_id, a_id))

        pattern = self.behavior_ctrl.select_pattern(
            tick, self.config.num_ticks, rel, self.rng
        )

        dialog_id = f"dlg_{tick}_{uuid.uuid4().hex[:6]}"
        dialog = DialogData(
            dialog_id=dialog_id,
            tick=tick,
            participants=[a_id, b_id],
            conversation_pattern=pattern,
        )

        # 3-6 turns per conversation
        n_turns = self.rng.randint(3, 6)
        history: list[str] = []
        speakers = [synth_a, synth_b]
        speaker_ids = [a_id, b_id]

        color_a = SYNTH_COLORS[list(self.synths.keys()).index(a_id) % len(SYNTH_COLORS)]
        color_b = SYNTH_COLORS[list(self.synths.keys()).index(b_id) % len(SYNTH_COLORS)]
        colors = [color_a, color_b]

        for turn_idx in range(n_turns):
            if not self.state.within_budget():
                break

            idx = turn_idx % 2
            speaker = speakers[idx]
            speaker_id = speaker_ids[idx]
            partner = speakers[1 - idx]
            cognitive = self.cognitive_states[speaker_id]
            ego = self.ego_states.get(speaker_id)

            # Build system prompt via Tessera-CT
            synth_events = [e for e in events if speaker_id in e.affected_synths]
            system_prompt = self.tessera.assemble(
                synth=speaker,
                cognitive=cognitive,
                events=synth_events,
                tick=tick,
                total_ticks=self.config.num_ticks,
                setting=self.config.context_description[:200],
                relationship=rel,
                partner_name=partner.name,
                inner_monologue=ego.inner_monologue if ego else "",
                goals=ego.goals if ego else None,
            )

            # Map personality to LLM params
            position = self._pattern_position(pattern, turn_idx)
            params = self.param_mapper.map(
                speaker, cognitive, position, intensity
            )

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            for i, line in enumerate(history):
                role = "assistant" if i % 2 == idx else "user"
                messages.append({"role": role, "content": line})

            # Generate response
            response = self.client.call(
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )

            if isinstance(response, dict) and "error" in response:
                response = "..."

            # Clean response (strip actions, name prefixes)
            response = re.sub(r"\*.*?\*", "", str(response))
            response = re.sub(r"^.*?:", "", response).strip() or "..."

            # Print colored dialog
            console.print(
                f"  [{colors[idx]}]{speaker.name}:[/{colors[idx]}] {response}"
            )

            # Record turn
            turn = DialogTurn(
                speaker=speaker_id,
                content=response,
                emotional_tone=cognitive.emotional_valence > 0.2
                and "positive" or "neutral",
                inner_monologue=ego.inner_monologue if ego else None,
                llm_params={
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "max_tokens": params.max_tokens,
                },
            )
            dialog.turns.append(turn)
            history.append(response)

        # Update relationship based on conversation
        if rel:
            conversation_summary = "\n".join(
                f"{self.synths[t.speaker].name}: {t.content}" for t in dialog.turns
            )
            rel = self.rel_gen.update_relationship(
                rel, synth_a, synth_b, conversation_summary
            )
            key = (a_id, b_id) if (a_id, b_id) in self.relationships else (b_id, a_id)
            self.relationships[key] = rel

        # Persist
        turns_json = json.dumps([t.model_dump(mode="json") for t in dialog.turns])
        self.store.save_dialog(
            self.config.name, dialog_id, tick,
            [a_id, b_id], turns_json, pattern,
        )

        return dialog

    def _pattern_position(self, pattern: str, turn_idx: int) -> str:
        """Map pattern + turn index to a token allocation position."""
        if pattern == "BANTER":
            return "banter"
        elif pattern == "DEBATE":
            return "debate_turn"
        elif pattern == "ARGUMENT":
            positions = ["argument_claim", "argument_rebuttal", "argument_claim", "argument_rebuttal"]
            return positions[turn_idx % len(positions)]
        elif pattern == "CONFESSION":
            return "confession" if turn_idx == 0 else "brief_reaction"
        elif pattern == "STORYTELLING":
            return "storytelling" if turn_idx == 1 else "brief_reaction"
        elif pattern == "INTERROGATION":
            return "question" if turn_idx % 2 == 0 else "answer"
        return "standard"

    def _init_cognitive_states(self):
        """Initialize cognitive tensors for all synths."""
        for synth_id, synth in self.synths.items():
            baseline = synth.psychographics.emotional_baseline
            valence = {
                "cheerful": 0.4, "calm": 0.1, "neutral": 0.0,
                "anxious": -0.2, "melancholic": -0.3, "irritable": -0.2,
                "stoic": 0.0,
            }.get(baseline, 0.0)
            arousal = {
                "cheerful": 0.4, "calm": 0.1, "neutral": 0.2,
                "anxious": 0.6, "melancholic": 0.2, "irritable": 0.5,
                "stoic": 0.1,
            }.get(baseline, 0.2)

            self.cognitive_states[synth_id] = CognitiveTensor(
                emotional_valence=valence,
                emotional_arousal=arousal,
                energy_budget=80.0,
                social_engagement=synth.psychographics.big_five.extraversion,
            )

    def _apply_event_impacts(self, event: SimEvent):
        """Apply event impacts to cognitive states."""
        for impact in event.impacts:
            if impact.synth_id in self.cognitive_states:
                cog = self.cognitive_states[impact.synth_id]
                self.cognitive_states[impact.synth_id] = CognitiveTensor(
                    knowledge_state=cog.knowledge_state + impact.knowledge_gained,
                    emotional_valence=max(-1.0, min(1.0, cog.emotional_valence + impact.emotional_valence_delta)),
                    emotional_arousal=max(0.0, min(1.0, cog.emotional_arousal + impact.emotional_arousal_delta)),
                    energy_budget=max(0.0, min(100.0, cog.energy_budget + impact.energy_delta)),
                    decision_confidence=cog.decision_confidence,
                    patience_threshold=cog.patience_threshold,
                    risk_tolerance=cog.risk_tolerance,
                    social_engagement=cog.social_engagement,
                )

    def _get_recent_dialog_summary(self, synth_id: str) -> str:
        """Get a brief summary of recent dialogs involving a synth."""
        recent = [d for d in self.all_dialogs[-5:] if synth_id in d.participants]
        if not recent:
            return ""
        lines = []
        for d in recent[-2:]:
            for t in d.turns[-3:]:
                name = self.synths.get(t.speaker, SynthProfile(
                    synth_id=t.speaker, name=t.speaker,
                    demographics={"age": 0, "gender": "", "occupation": ""}
                )).name
                lines.append(f"{name}: {t.content[:60]}")
        return "\n".join(lines)

    def _persist_synths(self):
        """Save all synth profiles to storage."""
        for synth_id, synth in self.synths.items():
            self.store.save_synth(
                self.config.name,
                synth_id,
                synth.name,
                synth.model_dump_json(),
            )

    def _persist_relationships(self, tick: int):
        """Save all relationship states to storage."""
        for (a, b), rel in self.relationships.items():
            self.store.save_relationship(
                self.config.name, a, b, tick,
                rel.metrics.model_dump_json(),
                rel.description, rel.relationship_type,
            )

    def _print_graph_summary(self):
        """Print graph summary."""
        G = self.graph
        console.print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        console.print(f"  Density: {nx.density(G):.3f}, Connected: {nx.is_connected(G)}")

    def _print_synth_summary(self):
        """Print generated synths."""
        for i, (sid, synth) in enumerate(self.synths.items()):
            color = SYNTH_COLORS[i % len(SYNTH_COLORS)]
            console.print(
                f"  [{color}]{synth.name}[/{color}] ({sid}): "
                f"{synth.demographics.age}yo {synth.demographics.occupation}"
            )
