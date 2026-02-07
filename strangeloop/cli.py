"""Click CLI entry point for Strange Loop."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from strangeloop.daity.engine import SimulationEngine
from strangeloop.output.reporter import generate_report
from strangeloop.output.exporter import export_graph
from strangeloop.schemas.simulation import SimulationConfig
from strangeloop.visualization import generate_dashboard

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Strange Loop: Synthetic social network simulator."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--output-dir", "-o", type=str, default=None, help="Override output directory")
def run(config_path: str, verbose: bool, output_dir: str | None):
    """Run a simulation from a config file."""
    config_data = json.loads(Path(config_path).read_text())
    if verbose:
        config_data["verbose"] = True
    if output_dir:
        config_data["output_dir"] = output_dir

    config = SimulationConfig(**config_data)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    engine = SimulationEngine(config)
    state = engine.run()

    # Generate outputs
    report_path = Path(config.output_dir) / f"{config.name}_report.md"
    generate_report(
        config=config,
        synths=engine.synths,
        relationships=engine.relationships,
        dialogs=engine.all_dialogs,
        events=engine.all_events,
        token_usage=state.token_usage,
        output_path=str(report_path),
    )
    console.print(f"\nReport: [link={report_path}]{report_path}[/link]")

    if engine.graph:
        graph_path = Path(config.output_dir) / f"{config.name}_graph"
        export_graph(engine.graph, str(graph_path))
        console.print(f"Graph: {graph_path}.json, {graph_path}.graphml")

    # Generate interactive dashboard
    dash_path = Path(config.output_dir) / f"{config.name}_dashboard.html"
    generate_dashboard(
        config=config,
        synths=engine.synths,
        relationships=engine.relationships,
        dialogs=engine.all_dialogs,
        events=engine.all_events,
        token_usage=state.token_usage,
        graph=engine.graph,
        output_path=str(dash_path),
    )
    console.print(f"Dashboard: [link=file://{dash_path.resolve()}]{dash_path}[/link]")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True))
def viz(output_dir: str):
    """Open a previously generated dashboard from an output directory."""
    import webbrowser

    output = Path(output_dir)
    dashboards = sorted(output.glob("*_dashboard.html"))
    if not dashboards:
        console.print(f"[red]No dashboard found in {output_dir}[/red]")
        console.print("Run a simulation first: strangeloop run <config>")
        sys.exit(1)

    path = dashboards[-1]
    console.print(f"Opening: {path}")
    webbrowser.open(f"file://{path.resolve()}")


@cli.command()
@click.argument("output_dir", type=click.Path(), default=".")
def init(output_dir: str):
    """Create an example configuration file."""
    example = {
        "name": "my_simulation",
        "description": "A social simulation",
        "seed": 42,
        "graph": {
            "num_synths": 5,
            "topology": "small_world",
            "edge_density": 0.3,
            "avg_connections": 3,
            "rewire_prob": 0.3,
            "num_communities": 2,
            "seed": 42,
        },
        "num_ticks": 3,
        "time_scale": "days",
        "start_date": "2025-06-01",
        "token_budget": 50000,
        "cost_limit_usd": 2.0,
        "default_model": "openai/gpt-4o",
        "fallback_model": "meta-llama/llama-3.1-70b-instruct:free",
        "synth_detail_level": "standard",
        "context_description": "A small neighborhood where everyone knows each other.",
        "envelope": {"attack": 0.2, "decay": 0.1, "sustain": 0.8, "release": 0.3},
        "conversation_patterns": ["DEBATE", "BANTER", "ARGUMENT"],
        "event_frequency": 0.3,
        "event_types": ["personal", "social"],
        "animism_mode": False,
        "output_dir": "output",
        "verbose": False,
    }

    output_path = Path(output_dir) / "strangeloop_config.json"
    output_path.write_text(json.dumps(example, indent=2))
    console.print(f"Created: {output_path}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def info(config_path: str):
    """Show simulation info without running it."""
    config_data = json.loads(Path(config_path).read_text())
    config = SimulationConfig(**config_data)

    from strangeloop.brancher.generator import generate_graph, print_graph_summary

    console.print(f"[bold]{config.name}[/bold]: {config.description}")
    console.print(f"  Ticks: {config.num_ticks}")
    console.print(f"  Model: {config.default_model}")
    console.print(f"  Budget: {config.token_budget:,} tokens (${config.cost_limit_usd:.2f})")
    console.print(f"  Graph: {config.graph.num_synths} synths, {config.graph.topology.value} topology")

    console.print("\n[bold]Graph preview:[/bold]")
    graph = generate_graph(config.graph)
    print_graph_summary(graph)


if __name__ == "__main__":
    cli()
