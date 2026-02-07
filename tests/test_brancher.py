"""Tests for brancher (graph generation)."""

import networkx as nx
import pytest

from strangeloop.brancher.generator import (
    compute_centralities,
    generate_graph,
    nodes_by_centrality,
)
from strangeloop.schemas.simulation import GraphConfig, Topology


class TestGenerateGraph:
    def test_small_world(self):
        config = GraphConfig(num_synths=10, topology=Topology.SMALL_WORLD, seed=42)
        G = generate_graph(config)
        assert G.number_of_nodes() == 10
        assert nx.is_connected(G)
        assert all(n.startswith("synth_") for n in G.nodes())

    def test_scale_free(self):
        config = GraphConfig(num_synths=10, topology=Topology.SCALE_FREE, seed=42)
        G = generate_graph(config)
        assert G.number_of_nodes() == 10
        assert nx.is_connected(G)

    def test_community(self):
        config = GraphConfig(
            num_synths=12, topology=Topology.COMMUNITY,
            num_communities=3, seed=42,
        )
        G = generate_graph(config)
        assert G.number_of_nodes() == 12
        assert nx.is_connected(G)

    def test_random(self):
        config = GraphConfig(
            num_synths=10, topology=Topology.RANDOM,
            edge_density=0.5, seed=42,
        )
        G = generate_graph(config)
        assert G.number_of_nodes() == 10
        assert nx.is_connected(G)

    def test_centralities_stored(self):
        config = GraphConfig(num_synths=8, topology=Topology.SMALL_WORLD, seed=42)
        G = generate_graph(config)
        for node in G.nodes():
            assert "eigenvector" in G.nodes[node]
            assert "betweenness" in G.nodes[node]
            assert "pagerank" in G.nodes[node]
            assert "degree" in G.nodes[node]

    def test_deterministic_with_seed(self):
        config = GraphConfig(num_synths=10, topology=Topology.SMALL_WORLD, seed=42)
        G1 = generate_graph(config)
        G2 = generate_graph(config)
        assert set(G1.edges()) == set(G2.edges())


class TestCentralities:
    def test_compute(self):
        G = nx.watts_strogatz_graph(10, 4, 0.3, seed=42)
        c = compute_centralities(G)
        assert "eigenvector" in c
        assert "betweenness" in c
        assert "pagerank" in c
        assert "degree" in c
        assert len(c["eigenvector"]) == 10


class TestNodesByCentrality:
    def test_ordering(self):
        config = GraphConfig(num_synths=10, topology=Topology.SCALE_FREE, seed=42)
        G = generate_graph(config)
        ordered = nodes_by_centrality(G)
        assert len(ordered) == 10
        scores = [G.nodes[n]["eigenvector"] for n in ordered]
        assert scores == sorted(scores, reverse=True)
