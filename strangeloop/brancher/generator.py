"""Social graph generation with multiple topology options.

Adapted from timepoint-daedalus/graph.py.
"""

import warnings
from typing import Optional

import networkx as nx
import numpy as np

from strangeloop.schemas.simulation import GraphConfig, Topology


def generate_graph(config: GraphConfig) -> nx.Graph:
    """Generate a social network graph from configuration.

    Supports: small_world (Watts-Strogatz), scale_free (Barabasi-Albert),
    community (stochastic block model), random (Erdos-Renyi).
    """
    seed = config.seed
    n = config.num_synths
    k = min(config.avg_connections, n - 1)

    if config.topology == Topology.SMALL_WORLD:
        # Watts-Strogatz: clustered with short path lengths
        k_even = max(2, k if k % 2 == 0 else k + 1)
        G = nx.watts_strogatz_graph(n, k_even, config.rewire_prob, seed=seed)

    elif config.topology == Topology.SCALE_FREE:
        # Barabasi-Albert: preferential attachment, hub-and-spoke
        m = max(1, k // 2)
        G = nx.barabasi_albert_graph(n, m, seed=seed)

    elif config.topology == Topology.COMMUNITY:
        # Stochastic block model: distinct communities
        nc = config.num_communities
        sizes = [n // nc] * nc
        sizes[-1] += n - sum(sizes)  # remainder to last community
        # Dense within community, sparse between
        p_in = min(config.edge_density * 3, 0.9)
        p_out = config.edge_density * 0.3
        probs = [[p_in if i == j else p_out for j in range(nc)] for i in range(nc)]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)

    elif config.topology == Topology.RANDOM:
        # Erdos-Renyi
        G = nx.erdos_renyi_graph(n, config.edge_density, seed=seed)

    else:
        G = nx.watts_strogatz_graph(n, max(2, k), 0.3, seed=seed)

    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = list(components[i - 1])[0]
            v = list(components[i])[0]
            G.add_edge(u, v)

    # Label nodes with synth IDs
    mapping = {i: f"synth_{i}" for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Compute and store centralities
    centralities = compute_centralities(G)
    for node in G.nodes():
        G.nodes[node]["eigenvector"] = centralities["eigenvector"].get(node, 0.0)
        G.nodes[node]["betweenness"] = centralities["betweenness"].get(node, 0.0)
        G.nodes[node]["pagerank"] = centralities["pagerank"].get(node, 0.0)
        G.nodes[node]["degree"] = centralities["degree"].get(node, 0)

    return G


def compute_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute centrality metrics for all nodes."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = {
            "eigenvector": {},
            "betweenness": nx.betweenness_centrality(G),
            "pagerank": nx.pagerank(G),
            "degree": {n: d for n, d in G.degree()},
        }
        try:
            result["eigenvector"] = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            result["eigenvector"] = nx.degree_centrality(G)
        return result


def nodes_by_centrality(G: nx.Graph, metric: str = "eigenvector") -> list[str]:
    """Return node IDs sorted by centrality (most central first)."""
    scores = {n: G.nodes[n].get(metric, 0.0) for n in G.nodes()}
    return sorted(scores, key=scores.get, reverse=True)


def print_graph_summary(G: nx.Graph):
    """Print ASCII summary of graph structure."""
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold]Graph Summary[/bold]")
    console.print(f"  Nodes: {G.number_of_nodes()}")
    console.print(f"  Edges: {G.number_of_edges()}")
    console.print(f"  Density: {nx.density(G):.4f}")
    console.print(f"  Connected: {nx.is_connected(G)}")

    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        max_node = max(degrees, key=degrees.get)
        console.print(f"  Avg degree: {avg_degree:.2f}")
        console.print(f"  Most connected: {max_node} ({degrees[max_node]})")
