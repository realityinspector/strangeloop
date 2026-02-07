"""LLM-powered synthetic persona generation.

Generates synths in centrality order so important characters
inform neighbor generation.
"""

import networkx as nx
from rich.console import Console

from strangeloop.brancher.generator import nodes_by_centrality
from strangeloop.gpt_genius.client import LLMClient
from strangeloop.schemas.synth import (
    Backstory, BigFive, Demographics, Psychographics,
    SocialMediaBehavior, SynthProfile,
)
from strangeloop.synthgen.prompts import SYNTH_GENERATION_SYSTEM, SYNTH_GENERATION_USER

console = Console()


class SynthGenerator:
    """Generate synth profiles using LLM calls in centrality order."""

    def __init__(self, client: LLMClient, detail_level: str = "standard"):
        self.client = client
        self.detail_level = detail_level

    def generate_all(
        self,
        graph: nx.Graph,
        context: str,
    ) -> dict[str, SynthProfile]:
        """Generate synth profiles for all nodes, most-central first."""
        ordered_nodes = nodes_by_centrality(graph)
        synths: dict[str, SynthProfile] = {}

        for rank, node_id in enumerate(ordered_nodes, 1):
            console.print(f"  [dim]Generating synth {rank}/{len(ordered_nodes)}: {node_id}[/dim]")

            neighbor_context = self._build_neighbor_context(graph, node_id, synths)
            node_data = graph.nodes[node_id]

            user_prompt = SYNTH_GENERATION_USER.format(
                context=context,
                node_id=node_id,
                centrality_rank=rank,
                total_nodes=len(ordered_nodes),
                eigenvector=node_data.get("eigenvector", 0.0),
                neighbor_count=graph.degree(node_id),
                detail_level=self.detail_level,
                neighbor_context=neighbor_context,
            )

            result = self.client.call(
                messages=[
                    {"role": "system", "content": SYNTH_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=800,
                json_mode=True,
            )

            profile = self._parse_profile(result, node_id, node_data)
            synths[node_id] = profile

        return synths

    def _build_neighbor_context(
        self, graph: nx.Graph, node_id: str, existing_synths: dict[str, SynthProfile]
    ) -> str:
        """Build context string from already-generated neighbors."""
        neighbors = list(graph.neighbors(node_id))
        generated = [n for n in neighbors if n in existing_synths]

        if not generated:
            return "No neighbors generated yet."

        lines = ["Already-generated neighbors:"]
        for nid in generated:
            s = existing_synths[nid]
            lines.append(f"- {s.name} ({nid}): {s.demographics.occupation}, {s.voice_description[:80]}")
        return "\n".join(lines)

    def _parse_profile(self, result: dict, node_id: str, node_data: dict) -> SynthProfile:
        """Parse LLM JSON result into a SynthProfile."""
        if isinstance(result, dict) and "error" in result:
            return self._fallback_profile(node_id, node_data)

        try:
            demo_data = result.get("demographics", {})
            demographics = Demographics(
                age=demo_data.get("age", 30),
                gender=demo_data.get("gender", "unknown"),
                occupation=demo_data.get("occupation", "unknown"),
                education_level=demo_data.get("education_level", ""),
                location=demo_data.get("location", ""),
                ethnicity=demo_data.get("ethnicity", ""),
                socioeconomic_status=demo_data.get("socioeconomic_status", ""),
            )

            psych_data = result.get("psychographics", {})
            bf_data = psych_data.get("big_five", {})
            psychographics = Psychographics(
                big_five=BigFive(
                    openness=bf_data.get("openness", 0.5),
                    conscientiousness=bf_data.get("conscientiousness", 0.5),
                    extraversion=bf_data.get("extraversion", 0.5),
                    agreeableness=bf_data.get("agreeableness", 0.5),
                    neuroticism=bf_data.get("neuroticism", 0.5),
                ),
                values=psych_data.get("values", []),
                interests=psych_data.get("interests", []),
                communication_style=psych_data.get("communication_style", "conversational"),
                vocabulary_complexity=psych_data.get("vocabulary_complexity", "standard"),
                emotional_baseline=psych_data.get("emotional_baseline", "neutral"),
            )

            back_data = result.get("backstory", {})
            backstory = Backstory(
                summary=back_data.get("summary", ""),
                key_events=back_data.get("key_events", []),
                motivations=back_data.get("motivations", []),
                fears=back_data.get("fears", []),
                secrets=back_data.get("secrets", []),
            )

            social_data = result.get("social_behavior", {})
            social = SocialMediaBehavior(
                conflict_style=social_data.get("conflict_style", "avoidant"),
                influence_seeking=social_data.get("influence_seeking", 0.5),
                group_conformity=social_data.get("group_conformity", 0.5),
            )

            return SynthProfile(
                synth_id=node_id,
                name=result.get("name", f"Synth {node_id}"),
                demographics=demographics,
                psychographics=psychographics,
                backstory=backstory,
                social_behavior=social,
                voice_description=result.get("voice_description", ""),
                eigenvector_centrality=node_data.get("eigenvector", 0.0),
                node_id=node_id,
            )
        except Exception:
            return self._fallback_profile(node_id, node_data)

    def _fallback_profile(self, node_id: str, node_data: dict) -> SynthProfile:
        """Create a minimal fallback profile when LLM fails."""
        return SynthProfile(
            synth_id=node_id,
            name=f"Synth {node_id}",
            demographics=Demographics(age=30, gender="unknown", occupation="unknown"),
            eigenvector_centrality=node_data.get("eigenvector", 0.0),
            node_id=node_id,
        )
