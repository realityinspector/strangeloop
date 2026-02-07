# Strange Loop

A synthetic social network simulator for RLSF (Reinforcement Learning from Social Feedback). Generates populations of AI personas with distinct personalities, places them in a social graph, and simulates their conversations, relationships, and reactions to events over time.

## Architecture

```
strangeloop/
  brancher/        Graph topology generation (NetworkX)
  synthgen/        Synthetic persona generation via LLM
  relgen/          Relationship initialization and evolution
  eventgen/        Stochastic event generation
  ego_loop/        Per-synth inner monologue and goal tracking
  tessera_ct/      System prompt assembly (Conversational Tensor)
  persona2parameters/  Personality -> LLM sampling params mapping
  knobs/           ADSR envelope + behavior pattern controller
  gpt_genius/      OpenRouter LLM client with retry logic
  schemas/         Pydantic data models (synths, relationships, events, dialogs)
  storage/         SQLite persistence
  output/          Markdown report + graph export (JSON, GraphML)
  visualization/   Interactive HTML dashboard generation
```

## Quickstart

```bash
# Install
uv pip install -e .

# Create example config
strangeloop init

# Run simulation
strangeloop run strangeloop_config.json

# Open dashboard in browser
strangeloop viz output
```

## Commands

| Command | Description |
|---------|-------------|
| `strangeloop run <config>` | Run a simulation and generate report, graph, and dashboard |
| `strangeloop viz <output_dir>` | Open a previously generated dashboard in the browser |
| `strangeloop init [dir]` | Create an example configuration file |
| `strangeloop info <config>` | Preview simulation settings without running |

## Visualization Dashboard

After a simulation completes, a self-contained HTML dashboard is generated alongside the markdown report. It includes:

- **Network Graph** -- Interactive D3.js force-directed graph. Nodes sized by eigenvector centrality, colored by dominant Big Five trait. Edges show trust (thickness) and emotional bond (color). Drag nodes, hover for details, click to jump to character cards.
- **Character Cards** -- Grid of synth profiles with avatar, demographics, Big Five radar chart (SVG), voice description, backstory, and value/interest tags.
- **Relationship Matrix** -- NxN heatmap colored by trust level (red=low, green=high). Hover any cell for full metrics: trust, bond, power dynamic, belief alignment.
- **Event Timeline** -- Chronological event cards grouped by tick, with type badges, severity indicators, and affected synths.
- **ADSR Envelope** -- SVG line chart showing the simulation's intensity curve with shaded Attack/Decay/Sustain/Release phases and tick markers.
- **Conversation Log** -- Collapsible dialog sections per tick with colored speaker names, conversation pattern badges, and LLM parameter pills (temperature, top_p, max_tokens).

The dashboard is a single HTML file with D3.js loaded from CDN. No build step, no extra dependencies. Open it in any browser.

## Configuration

See `config/examples/small_town.json` for a full example. Key settings:

```json
{
  "name": "small_town",
  "graph": { "num_synths": 8, "topology": "community" },
  "num_ticks": 5,
  "default_model": "openai/gpt-4o",
  "envelope": { "attack": 0.3, "decay": 0.1, "sustain": 0.8, "release": 0.2 },
  "conversation_patterns": ["DEBATE", "ARGUMENT", "BANTER", "CONFESSION"],
  "event_frequency": 0.4
}
```

## Dependencies

- `openai` -- LLM API (via OpenRouter)
- `networkx` -- Social graph generation and analysis
- `pydantic` -- Data validation and serialization
- `sqlmodel` -- Database persistence
- `click` -- CLI framework
- `rich` -- Terminal output formatting
- `numpy` -- Numerical operations

No additional dependencies for visualization (D3.js loaded from CDN).

## Environment

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

## License

MIT
