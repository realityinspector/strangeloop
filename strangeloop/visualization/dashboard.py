"""Generate a self-contained HTML dashboard from simulation data.

Produces a single HTML file with embedded D3.js (CDN), inline CSS/JS,
and all simulation data as a JSON blob. No extra Python dependencies.
"""

import html
import json
from pathlib import Path

import networkx as nx

from strangeloop.schemas.dialog import DialogData
from strangeloop.schemas.event import SimEvent
from strangeloop.schemas.relationship import RelationshipState
from strangeloop.schemas.simulation import SimulationConfig, TokenUsage
from strangeloop.schemas.synth import SynthProfile


def _esc(text: str) -> str:
    """Escape a string for safe embedding in HTML."""
    return html.escape(str(text), quote=True)


def generate_dashboard(
    config: SimulationConfig,
    synths: dict[str, SynthProfile],
    relationships: dict[tuple[str, str], RelationshipState],
    dialogs: list[DialogData],
    events: list[SimEvent],
    token_usage: TokenUsage,
    graph: nx.Graph | None,
    output_path: str,
) -> str:
    """Generate a self-contained HTML dashboard.

    Returns the output path for convenience.
    """
    data = _collect_data(config, synths, relationships, dialogs, events, token_usage, graph)
    html_str = _render_template(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_str)
    return output_path


def _collect_data(
    config: SimulationConfig,
    synths: dict[str, SynthProfile],
    relationships: dict[tuple[str, str], RelationshipState],
    dialogs: list[DialogData],
    events: list[SimEvent],
    token_usage: TokenUsage,
    graph: nx.Graph | None,
) -> dict:
    """Serialize all simulation data into a single JSON-friendly dict.

    All string values that will be rendered in HTML are escaped here at
    the data layer so the JS side can safely use textContent / d3.text()
    without risk of script injection.
    """

    # -- Nodes & edges from the graph --
    nodes = []
    edges = []
    if graph:
        for node_id in graph.nodes():
            attrs = graph.nodes[node_id]
            synth = synths.get(node_id)
            big_five = None
            if synth:
                bf = synth.psychographics.big_five
                big_five = {
                    "openness": bf.openness,
                    "conscientiousness": bf.conscientiousness,
                    "extraversion": bf.extraversion,
                    "agreeableness": bf.agreeableness,
                    "neuroticism": bf.neuroticism,
                }
            dominant_trait = None
            if big_five:
                dominant_trait = max(big_five, key=big_five.get)

            nodes.append({
                "id": node_id,
                "name": _esc(synth.name) if synth else node_id,
                "occupation": _esc(synth.demographics.occupation) if synth else "",
                "age": synth.demographics.age if synth else 0,
                "eigenvector": attrs.get("eigenvector", 0.1),
                "betweenness": attrs.get("betweenness", 0.0),
                "big_five": big_five,
                "dominant_trait": dominant_trait,
            })

        for u, v in graph.edges():
            rel = relationships.get((u, v)) or relationships.get((v, u))
            edges.append({
                "source": u,
                "target": v,
                "trust": rel.metrics.trust_level if rel else 0.5,
                "bond": rel.metrics.emotional_bond if rel else 0.0,
                "type": _esc(rel.relationship_type) if rel else "acquaintance",
            })

    # -- Synth cards --
    synth_list = []
    for sid, s in synths.items():
        bf = s.psychographics.big_five
        synth_list.append({
            "id": sid,
            "name": _esc(s.name),
            "age": s.demographics.age,
            "gender": _esc(s.demographics.gender),
            "occupation": _esc(s.demographics.occupation),
            "education": _esc(s.demographics.education_level),
            "location": _esc(s.demographics.location),
            "big_five": {
                "openness": bf.openness,
                "conscientiousness": bf.conscientiousness,
                "extraversion": bf.extraversion,
                "agreeableness": bf.agreeableness,
                "neuroticism": bf.neuroticism,
            },
            "voice": _esc(s.voice_description),
            "backstory": _esc(s.backstory.summary),
            "values": [_esc(v) for v in s.psychographics.values[:5]],
            "interests": [_esc(v) for v in s.psychographics.interests[:5]],
            "communication_style": _esc(s.psychographics.communication_style),
            "emotional_baseline": _esc(s.psychographics.emotional_baseline),
            "conflict_style": _esc(s.social_behavior.conflict_style),
            "eigenvector": s.eigenvector_centrality,
        })

    # -- Relationship matrix --
    synth_ids = list(synths.keys())
    rel_matrix = []
    for (a, b), rel in relationships.items():
        rel_matrix.append({
            "synth_a": a,
            "synth_b": b,
            "name_a": _esc(synths[a].name) if a in synths else a,
            "name_b": _esc(synths[b].name) if b in synths else b,
            "trust": rel.metrics.trust_level,
            "bond": rel.metrics.emotional_bond,
            "power": rel.metrics.power_dynamic,
            "alignment": rel.metrics.belief_alignment,
            "type": _esc(rel.relationship_type),
            "description": _esc(rel.description),
        })

    # -- Events --
    event_list = []
    for e in events:
        event_list.append({
            "id": e.event_id,
            "tick": e.tick,
            "type": e.event_type.value,
            "title": _esc(e.title),
            "description": _esc(e.description),
            "severity": e.severity,
            "affected": [
                _esc(synths[s].name) if s in synths else s
                for s in e.affected_synths
            ],
        })

    # -- ADSR envelope curve (sample 100 points) --
    from strangeloop.knobs.envelope import EnvelopeConfig as KnobsEnvelope
    envelope_obj = KnobsEnvelope(
        attack=config.envelope.attack,
        decay=config.envelope.decay,
        sustain=config.envelope.sustain,
        release=config.envelope.release,
    )
    envelope_curve = []
    for i in range(101):
        p = i / 100.0
        envelope_curve.append({
            "progress": p,
            "intensity": envelope_obj.intensity_at(p),
        })

    tick_markers = []
    for t in range(1, config.num_ticks + 1):
        p = t / config.num_ticks
        tick_markers.append({
            "tick": t,
            "progress": p,
            "intensity": envelope_obj.intensity_at(p),
        })

    # -- Dialogs --
    dialog_list = []
    for d in dialogs:
        turns = []
        for t in d.turns:
            turns.append({
                "speaker_id": t.speaker,
                "speaker_name": _esc(
                    synths[t.speaker].name if t.speaker in synths else t.speaker
                ),
                "content": _esc(t.content),
                "tone": t.emotional_tone,
                "llm_params": t.llm_params,
            })
        dialog_list.append({
            "id": d.dialog_id,
            "tick": d.tick,
            "participants": [
                _esc(synths[p].name) if p in synths else p
                for p in d.participants
            ],
            "pattern": d.conversation_pattern,
            "turns": turns,
        })

    return {
        "config": {
            "name": _esc(config.name),
            "description": _esc(config.context_description),
            "num_ticks": config.num_ticks,
            "model": _esc(config.default_model),
            "topology": config.graph.topology.value,
            "num_synths": config.graph.num_synths,
        },
        "token_usage": {
            "total": token_usage.total_tokens,
            "prompt": token_usage.prompt_tokens,
            "completion": token_usage.completion_tokens,
            "cost": token_usage.estimated_cost_usd,
        },
        "envelope": {
            "attack": config.envelope.attack,
            "decay": config.envelope.decay,
            "sustain": config.envelope.sustain,
            "release": config.envelope.release,
            "curve": envelope_curve,
            "tick_markers": tick_markers,
        },
        "nodes": nodes,
        "edges": edges,
        "synths": synth_list,
        "relationships": rel_matrix,
        "events": event_list,
        "dialogs": dialog_list,
        "synth_ids": synth_ids,
    }


def _render_template(data: dict) -> str:
    """Inject data into the HTML template and return complete HTML string."""
    data_json = json.dumps(data, default=str)
    return _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


# ---------------------------------------------------------------------------
# HTML Template
#
# Security note: All data values are HTML-escaped in _collect_data() before
# JSON serialization. The JS rendering uses textContent and d3.text() for
# user-generated strings. The only HTML construction uses our own static
# markup combined with pre-escaped data values.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strange Loop Dashboard</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --purple: #bc8cff;
  --orange: #f0883e;
  --cyan: #39d2c0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }

/* Header */
.header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}
.header h1 { font-size: 20px; font-weight: 600; }
.header .meta { display: flex; gap: 16px; flex-wrap: wrap; }
.header .pill {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2px 12px;
  font-size: 12px;
  color: var(--text-dim);
}
.header .pill b { color: var(--text); }

/* Nav */
.nav {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  display: flex;
  gap: 0;
  overflow-x: auto;
}
.nav a {
  padding: 10px 16px;
  font-size: 13px;
  color: var(--text-dim);
  border-bottom: 2px solid transparent;
  white-space: nowrap;
}
.nav a:hover, .nav a.active {
  color: var(--text);
  border-bottom-color: var(--accent);
}

/* Sections */
.section { padding: 24px; display: none; }
.section.active { display: block; }
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

/* Network Graph */
#graph-container {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}
#graph-container svg { display: block; }
.graph-tooltip {
  position: absolute;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 12px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s;
  z-index: 10;
}
.graph-tooltip.visible { opacity: 1; }

/* Legend */
.legend {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  flex-wrap: wrap;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-dim);
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

/* Character Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: 700;
  color: var(--bg);
  flex-shrink: 0;
}
.card-name { font-size: 16px; font-weight: 600; }
.card-subtitle { font-size: 12px; color: var(--text-dim); }
.card-row {
  font-size: 13px;
  color: var(--text-dim);
  margin-bottom: 4px;
}
.card-row b { color: var(--text); font-weight: 500; }
.radar-container { margin: 12px auto; }
.tag {
  display: inline-block;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 11px;
  color: var(--text-dim);
  margin: 2px;
}

/* Relationship Matrix */
.matrix-container { overflow-x: auto; position: relative; }
.matrix-container svg text { fill: var(--text-dim); }

/* Events */
.event-timeline {
  position: relative;
  padding-left: 24px;
}
.event-timeline::before {
  content: '';
  position: absolute;
  left: 11px;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--border);
}
.event-card {
  position: relative;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 12px;
}
.event-card::before {
  content: '';
  position: absolute;
  left: -19px;
  top: 16px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent);
  border: 2px solid var(--bg);
}
.event-badge {
  display: inline-block;
  border-radius: 4px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}
.event-badge.personal { background: #1f3d2a; color: var(--green); }
.event-badge.social { background: #2a1f3d; color: var(--purple); }
.event-badge.economic { background: #3d2a1f; color: var(--orange); }
.event-badge.environmental { background: #1f2a3d; color: var(--cyan); }
.severity-bar {
  display: inline-block;
  height: 4px;
  border-radius: 2px;
  background: var(--accent);
  vertical-align: middle;
}

/* ADSR */
#adsr-container {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
#adsr-container svg text { fill: var(--text-dim); font-size: 11px; }

/* Conversation Log */
.convo-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
}
.convo-header {
  padding: 10px 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
}
.convo-header:hover { background: var(--surface2); }
.convo-header .chevron {
  transition: transform 0.2s;
  color: var(--text-dim);
  font-size: 12px;
}
.convo-header.open .chevron { transform: rotate(90deg); }
.convo-body { display: none; padding: 0 16px 12px; }
.convo-body.open { display: block; }
.turn {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.turn:last-child { border-bottom: none; }
.turn .speaker { font-weight: 600; }
.turn .params {
  margin-top: 4px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.param-pill {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0px 6px;
  font-size: 10px;
  color: var(--text-dim);
  font-family: 'SF Mono', 'Cascadia Code', monospace;
}
.pattern-badge {
  display: inline-block;
  border-radius: 4px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 600;
  background: var(--surface2);
  border: 1px solid var(--accent);
  color: var(--accent);
}

/* Scroll-to-top */
.scroll-top {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: var(--accent);
  color: var(--bg);
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 18px;
  opacity: 0.8;
  border: none;
}
.scroll-top:hover { opacity: 1; }
</style>
</head>
<body>

<div class="header">
  <h1 id="sim-name"></h1>
  <div class="meta" id="header-meta"></div>
</div>

<nav class="nav" id="nav"></nav>

<div id="sections">
  <div class="section active" id="sec-network">
    <h2 class="section-title">Network Graph</h2>
    <div id="graph-container">
      <div class="graph-tooltip" id="graph-tooltip"></div>
    </div>
    <div class="legend" id="graph-legend"></div>
  </div>

  <div class="section" id="sec-characters">
    <h2 class="section-title">Characters</h2>
    <div class="card-grid" id="card-grid"></div>
  </div>

  <div class="section" id="sec-relationships">
    <h2 class="section-title">Relationship Matrix</h2>
    <div class="matrix-container" id="matrix-container"></div>
  </div>

  <div class="section" id="sec-events">
    <h2 class="section-title">Event Timeline</h2>
    <div class="event-timeline" id="event-timeline"></div>
  </div>

  <div class="section" id="sec-adsr">
    <h2 class="section-title">ADSR Intensity Envelope</h2>
    <div id="adsr-container"></div>
  </div>

  <div class="section" id="sec-conversations">
    <h2 class="section-title">Conversation Log</h2>
    <div id="convo-log"></div>
  </div>
</div>

<button class="scroll-top" onclick="window.scrollTo({top:0,behavior:'smooth'})">&#8593;</button>

<script>
// ---------------------------------------------------------------------------
// Utility: safe DOM helpers (all user-facing text uses textContent)
// ---------------------------------------------------------------------------
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) Object.entries(attrs).forEach(([k,v]) => {
    if (k === 'text') e.textContent = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k === 'className') e.className = v;
    else e.setAttribute(k, v);
  });
  if (children) children.forEach(c => { if (c) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
  return e;
}

// ---------------------------------------------------------------------------
// DATA (injected by Python — all string values are pre-escaped)
// ---------------------------------------------------------------------------
const DATA = __DATA_JSON__;

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------
const TRAIT_COLORS = {
  openness: '#58a6ff',
  conscientiousness: '#3fb950',
  extraversion: '#d29922',
  agreeableness: '#bc8cff',
  neuroticism: '#f85149',
};
const SYNTH_COLORS = [
  '#58a6ff','#3fb950','#d29922','#bc8cff','#f85149',
  '#f0883e','#39d2c0','#db61a2','#79c0ff','#7ee787',
];
function synthColor(idx) { return SYNTH_COLORS[idx % SYNTH_COLORS.length]; }
function nameToColor(name) {
  const idx = DATA.synths.findIndex(s => s.name === name);
  return idx >= 0 ? synthColor(idx) : '#8b949e';
}

// ---------------------------------------------------------------------------
// Header (safe: uses textContent via el())
// ---------------------------------------------------------------------------
document.getElementById('sim-name').textContent = DATA.config.name;
const meta = document.getElementById('header-meta');
const pills = [
  ['Ticks', String(DATA.config.num_ticks)],
  ['Model', DATA.config.model],
  ['Topology', DATA.config.topology],
  ['Tokens', DATA.token_usage.total.toLocaleString()],
  ['Cost', '$' + DATA.token_usage.cost.toFixed(2)],
];
pills.forEach(([label, val]) => {
  const pill = el('span', {className: 'pill'}, [
    document.createTextNode(label + ': '),
    el('b', {text: val}),
  ]);
  meta.appendChild(pill);
});

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
const TABS = [
  { id: 'sec-network', label: 'Network' },
  { id: 'sec-characters', label: 'Characters' },
  { id: 'sec-relationships', label: 'Relationships' },
  { id: 'sec-events', label: 'Events' },
  { id: 'sec-adsr', label: 'ADSR Envelope' },
  { id: 'sec-conversations', label: 'Conversations' },
];
const nav = document.getElementById('nav');
TABS.forEach((tab, i) => {
  const a = el('a', {href: '#', className: i === 0 ? 'active' : '', text: tab.label});
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav a').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.section').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    document.getElementById(tab.id).classList.add('active');
  });
  nav.appendChild(a);
});

// ---------------------------------------------------------------------------
// 1. Force-directed Network Graph (D3 — uses .text() which is safe)
// ---------------------------------------------------------------------------
(function() {
  const container = document.getElementById('graph-container');
  const width = container.clientWidth || 800;
  const height = 500;

  const svg = d3.select('#graph-container')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const tooltip = document.getElementById('graph-tooltip');

  const nodes = DATA.nodes.map(d => ({...d}));
  const links = DATA.edges.map(d => ({...d}));

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 4));

  const link = svg.append('g')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('stroke', d => {
      const bond = d.bond || 0;
      if (bond > 0.2) return '#3fb950';
      if (bond < -0.2) return '#f85149';
      return '#30363d';
    })
    .attr('stroke-width', d => Math.max(1, (d.trust || 0.5) * 4))
    .attr('stroke-opacity', 0.6);

  const node = svg.append('g')
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => d.dominant_trait ? TRAIT_COLORS[d.dominant_trait] : '#8b949e')
    .attr('stroke', '#0d1117')
    .attr('stroke-width', 2)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag', dragging)
      .on('end', dragEnd));

  const label = svg.append('g')
    .selectAll('text')
    .data(nodes)
    .join('text')
    .text(d => d.name)
    .attr('font-size', 11)
    .attr('fill', '#e6edf3')
    .attr('text-anchor', 'middle')
    .attr('dy', d => nodeRadius(d) + 14)
    .style('pointer-events', 'none');

  node.on('mouseover', function(event, d) {
    tooltip.textContent = '';
    tooltip.appendChild(el('b', {text: d.name}));
    tooltip.appendChild(document.createElement('br'));
    tooltip.appendChild(document.createTextNode(d.occupation));
    tooltip.appendChild(document.createElement('br'));
    tooltip.appendChild(document.createTextNode('Centrality: ' + d.eigenvector.toFixed(3)));
    tooltip.style.left = (event.offsetX + 12) + 'px';
    tooltip.style.top = (event.offsetY - 10) + 'px';
    tooltip.classList.add('visible');
  }).on('mouseout', () => {
    tooltip.classList.remove('visible');
  }).on('click', (event, d) => {
    document.querySelectorAll('.nav a').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.section').forEach(x => x.classList.remove('active'));
    nav.children[1].classList.add('active');
    document.getElementById('sec-characters').classList.add('active');
    const card = document.getElementById('card-' + d.id);
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node
      .attr('cx', d => d.x = Math.max(20, Math.min(width - 20, d.x)))
      .attr('cy', d => d.y = Math.max(20, Math.min(height - 20, d.y)));
    label
      .attr('x', d => d.x)
      .attr('y', d => d.y);
  });

  function nodeRadius(d) { return 8 + (d.eigenvector || 0.1) * 30; }

  function dragStart(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragging(event, d) { d.fx = event.x; d.fy = event.y; }
  function dragEnd(event, d) {
    if (!event.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  // Legend
  const legend = document.getElementById('graph-legend');
  Object.entries(TRAIT_COLORS).forEach(([trait, color]) => {
    const dot = el('span', {className: 'legend-dot', style: 'background:' + color});
    const item = el('span', {className: 'legend-item'}, [dot, document.createTextNode(trait)]);
    legend.appendChild(item);
  });
})();

// ---------------------------------------------------------------------------
// 2. Character Cards with Radar Charts (DOM-built, safe)
// ---------------------------------------------------------------------------
(function() {
  const grid = document.getElementById('card-grid');
  DATA.synths.forEach((s, idx) => {
    const card = el('div', {className: 'card', id: 'card-' + s.id});
    const initials = s.name.split(' ').map(w => w[0]).join('').slice(0, 2);
    const color = synthColor(idx);

    // Header
    const avatar = el('div', {className: 'avatar', style: 'background:' + color, text: initials});
    const nameEl = el('div', {className: 'card-name', text: s.name});
    const subtitle = el('div', {className: 'card-subtitle', text: s.age + 'yo \u00B7 ' + s.occupation});
    const headerInfo = el('div', {}, [nameEl, subtitle]);
    card.appendChild(el('div', {className: 'card-header'}, [avatar, headerInfo]));

    // Details
    const row1 = el('div', {className: 'card-row'});
    row1.appendChild(el('b', {text: 'Gender: '})); row1.appendChild(document.createTextNode(s.gender + ' \u00B7 '));
    row1.appendChild(el('b', {text: 'Education: '})); row1.appendChild(document.createTextNode(s.education || 'N/A'));
    card.appendChild(row1);

    const row2 = el('div', {className: 'card-row'});
    row2.appendChild(el('b', {text: 'Style: '})); row2.appendChild(document.createTextNode(s.communication_style + ' \u00B7 '));
    row2.appendChild(el('b', {text: 'Conflict: '})); row2.appendChild(document.createTextNode(s.conflict_style));
    card.appendChild(row2);

    const row3 = el('div', {className: 'card-row'});
    row3.appendChild(el('b', {text: 'Baseline: '})); row3.appendChild(document.createTextNode(s.emotional_baseline));
    card.appendChild(row3);

    // Radar chart container
    const radarDiv = el('div', {className: 'radar-container', id: 'radar-' + s.id});
    card.appendChild(radarDiv);

    // Voice
    if (s.voice) {
      const voiceRow = el('div', {className: 'card-row'});
      voiceRow.appendChild(el('b', {text: 'Voice: '}));
      voiceRow.appendChild(document.createTextNode(s.voice));
      card.appendChild(voiceRow);
    }

    // Backstory
    if (s.backstory) {
      const bs = s.backstory.length > 200 ? s.backstory.slice(0, 200) + '...' : s.backstory;
      card.appendChild(el('div', {className: 'card-row', style: 'margin-top:8px;font-size:12px;color:var(--text-dim)', text: bs}));
    }

    // Tags
    const allTags = (s.values || []).concat(s.interests || []);
    if (allTags.length > 0) {
      const tagContainer = el('div', {style: 'margin-top:8px'});
      allTags.forEach(t => tagContainer.appendChild(el('span', {className: 'tag', text: t})));
      card.appendChild(tagContainer);
    }

    grid.appendChild(card);
    drawRadar(s.big_five, '#radar-' + s.id, color);
  });

  function drawRadar(bf, selector, color) {
    const traits = ['openness','conscientiousness','extraversion','agreeableness','neuroticism'];
    const labels = ['O','C','E','A','N'];
    const values = traits.map(t => bf[t] || 0);
    const size = 120, cx = size/2, cy = size/2, r = 45;

    const svg = d3.select(selector)
      .append('svg')
      .attr('width', size)
      .attr('height', size);

    [0.25, 0.5, 0.75, 1.0].forEach(level => {
      svg.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', r * level)
        .attr('fill', 'none').attr('stroke', '#30363d').attr('stroke-width', 0.5);
    });

    const angleSlice = (Math.PI * 2) / traits.length;
    traits.forEach((_, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      svg.append('line')
        .attr('x1', cx).attr('y1', cy)
        .attr('x2', cx + r * Math.cos(angle)).attr('y2', cy + r * Math.sin(angle))
        .attr('stroke', '#30363d').attr('stroke-width', 0.5);
      svg.append('text')
        .attr('x', cx + (r + 12) * Math.cos(angle))
        .attr('y', cy + (r + 12) * Math.sin(angle))
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', '#8b949e').attr('font-size', 10)
        .text(labels[i]);
    });

    const points = values.map((v, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      return [cx + r * v * Math.cos(angle), cy + r * v * Math.sin(angle)];
    });
    svg.append('polygon')
      .attr('points', points.map(p => p.join(',')).join(' '))
      .attr('fill', color).attr('fill-opacity', 0.2)
      .attr('stroke', color).attr('stroke-width', 1.5);
    points.forEach(([x, y]) => {
      svg.append('circle').attr('cx', x).attr('cy', y).attr('r', 3).attr('fill', color);
    });
  }
})();

// ---------------------------------------------------------------------------
// 3. Relationship Heatmap (D3 — uses .text() which is safe)
// ---------------------------------------------------------------------------
(function() {
  const names = DATA.synths.map(s => s.name);
  const n = names.length;
  if (n < 2) return;

  const cellSize = Math.min(50, Math.max(30, 500 / n));
  const margin = { top: 80, left: 80, right: 20, bottom: 20 };
  const width = margin.left + n * cellSize + margin.right;
  const height = margin.top + n * cellSize + margin.bottom;

  const svg = d3.select('#matrix-container')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const trustLookup = {};
  const relLookup = {};
  DATA.relationships.forEach(r => {
    trustLookup[r.name_a + '|' + r.name_b] = r.trust;
    trustLookup[r.name_b + '|' + r.name_a] = r.trust;
    relLookup[r.name_a + '|' + r.name_b] = r;
    relLookup[r.name_b + '|' + r.name_a] = r;
  });

  const colorScale = d3.scaleLinear()
    .domain([0, 0.5, 1])
    .range(['#f85149', '#30363d', '#3fb950']);

  svg.selectAll('.col-label')
    .data(names).join('text').attr('class', 'col-label')
    .attr('x', (_, i) => margin.left + i * cellSize + cellSize / 2)
    .attr('y', margin.top - 8)
    .attr('text-anchor', 'start')
    .attr('transform', (_, i) => 'rotate(-45, ' + (margin.left + i * cellSize + cellSize / 2) + ', ' + (margin.top - 8) + ')')
    .attr('font-size', 11)
    .text(d => d);

  svg.selectAll('.row-label')
    .data(names).join('text').attr('class', 'row-label')
    .attr('x', margin.left - 8)
    .attr('y', (_, i) => margin.top + i * cellSize + cellSize / 2)
    .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
    .attr('font-size', 11)
    .text(d => d);

  // Matrix tooltip (DOM-built, safe)
  const matrixTooltip = document.createElement('div');
  matrixTooltip.style.cssText = 'position:absolute;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:12px;pointer-events:none;opacity:0;z-index:10';
  document.getElementById('matrix-container').appendChild(matrixTooltip);

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      if (row === col) {
        svg.append('rect')
          .attr('x', margin.left + col * cellSize).attr('y', margin.top + row * cellSize)
          .attr('width', cellSize - 1).attr('height', cellSize - 1)
          .attr('fill', '#161b22').attr('rx', 2);
        continue;
      }
      const key = names[row] + '|' + names[col];
      const trust = trustLookup[key];
      if (trust === undefined) {
        svg.append('rect')
          .attr('x', margin.left + col * cellSize).attr('y', margin.top + row * cellSize)
          .attr('width', cellSize - 1).attr('height', cellSize - 1)
          .attr('fill', '#0d1117').attr('rx', 2);
        continue;
      }
      const rel = relLookup[key];
      svg.append('rect')
        .attr('x', margin.left + col * cellSize).attr('y', margin.top + row * cellSize)
        .attr('width', cellSize - 1).attr('height', cellSize - 1)
        .attr('fill', colorScale(trust)).attr('rx', 2).attr('opacity', 0.8)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          d3.select(this).attr('opacity', 1).attr('stroke', '#e6edf3').attr('stroke-width', 1);
          matrixTooltip.textContent = '';
          matrixTooltip.appendChild(el('b', {text: names[row] + ' \u2194 ' + names[col]}));
          matrixTooltip.appendChild(document.createElement('br'));
          matrixTooltip.appendChild(document.createTextNode('Trust: ' + rel.trust.toFixed(2)));
          matrixTooltip.appendChild(document.createElement('br'));
          matrixTooltip.appendChild(document.createTextNode('Bond: ' + rel.bond.toFixed(2)));
          matrixTooltip.appendChild(document.createElement('br'));
          matrixTooltip.appendChild(document.createTextNode('Power: ' + rel.power.toFixed(2)));
          matrixTooltip.appendChild(document.createElement('br'));
          matrixTooltip.appendChild(document.createTextNode('Alignment: ' + rel.alignment.toFixed(2)));
          matrixTooltip.appendChild(document.createElement('br'));
          matrixTooltip.appendChild(document.createTextNode('Type: ' + rel.type));
          matrixTooltip.style.left = (event.offsetX + 12) + 'px';
          matrixTooltip.style.top = (event.offsetY - 10) + 'px';
          matrixTooltip.style.opacity = '1';
        })
        .on('mouseout', function() {
          d3.select(this).attr('opacity', 0.8).attr('stroke', 'none');
          matrixTooltip.style.opacity = '0';
        });
    }
  }
})();

// ---------------------------------------------------------------------------
// 4. Event Timeline (DOM-built, safe)
// ---------------------------------------------------------------------------
(function() {
  const container = document.getElementById('event-timeline');
  if (DATA.events.length === 0) {
    container.appendChild(el('div', {style: 'color:var(--text-dim);padding:20px', text: 'No events occurred during this simulation.'}));
    return;
  }

  const byTick = {};
  DATA.events.forEach(e => {
    if (!byTick[e.tick]) byTick[e.tick] = [];
    byTick[e.tick].push(e);
  });

  Object.keys(byTick).sort((a, b) => a - b).forEach(tick => {
    container.appendChild(el('div', {style: 'font-size:12px;color:var(--text-dim);margin:8px 0 4px;font-weight:600', text: 'Tick ' + tick}));

    byTick[tick].forEach(ev => {
      const card = el('div', {className: 'event-card'});

      // Badge row
      const badgeRow = el('div', {style: 'display:flex;align-items:center;gap:8px;margin-bottom:6px'});
      badgeRow.appendChild(el('span', {className: 'event-badge ' + ev.type, text: ev.type}));
      badgeRow.appendChild(el('span', {style: 'font-weight:600', text: ev.title}));
      const bar = el('span', {className: 'severity-bar'});
      bar.style.width = (ev.severity * 60) + 'px';
      bar.style.background = ev.severity > 0.7 ? 'var(--red)' : ev.severity > 0.4 ? 'var(--yellow)' : 'var(--green)';
      badgeRow.appendChild(bar);
      card.appendChild(badgeRow);

      // Description
      card.appendChild(el('div', {style: 'font-size:13px;color:var(--text-dim)', text: ev.description}));

      // Affected synths
      const affectedDiv = el('div', {style: 'margin-top:6px;font-size:12px'});
      affectedDiv.appendChild(document.createTextNode('Affected: '));
      ev.affected.forEach((name, i) => {
        if (i > 0) affectedDiv.appendChild(document.createTextNode(', '));
        affectedDiv.appendChild(el('span', {style: 'color:' + nameToColor(name) + ';font-weight:500', text: name}));
      });
      card.appendChild(affectedDiv);

      container.appendChild(card);
    });
  });
})();

// ---------------------------------------------------------------------------
// 5. ADSR Envelope Curve (D3 — uses .text() which is safe)
// ---------------------------------------------------------------------------
(function() {
  const container = document.getElementById('adsr-container');
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const width = (container.clientWidth || 700) - margin.left - margin.right;
  const height = 250;

  const svg = d3.select('#adsr-container')
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  const x = d3.scaleLinear().domain([0, 1]).range([0, width]);
  const y = d3.scaleLinear().domain([0, 1.1]).range([height, 0]);

  const env = DATA.envelope;
  const aEnd = env.attack * 0.25;
  const dEnd = aEnd + env.decay * 0.25;
  const rStart = 1.0 - env.release * 0.25;

  const phases = [
    { label: 'Attack', x0: 0, x1: aEnd, color: '#f8514920' },
    { label: 'Decay', x0: aEnd, x1: dEnd, color: '#d2992220' },
    { label: 'Sustain', x0: dEnd, x1: rStart, color: '#3fb95020' },
    { label: 'Release', x0: rStart, x1: 1, color: '#58a6ff20' },
  ];
  phases.forEach(p => {
    svg.append('rect')
      .attr('x', x(p.x0)).attr('y', 0)
      .attr('width', x(p.x1) - x(p.x0)).attr('height', height)
      .attr('fill', p.color);
    if (p.x1 - p.x0 > 0.02) {
      svg.append('text')
        .attr('x', x((p.x0 + p.x1) / 2)).attr('y', height - 8)
        .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', '#8b949e')
        .text(p.label);
    }
  });

  svg.append('g').attr('transform', 'translate(0,' + height + ')')
    .call(d3.axisBottom(x).ticks(10).tickFormat(d => (d * 100) + '%'))
    .selectAll('text').attr('fill', '#8b949e');
  svg.append('g').call(d3.axisLeft(y).ticks(5))
    .selectAll('text').attr('fill', '#8b949e');
  svg.selectAll('.domain, .tick line').attr('stroke', '#30363d');

  svg.append('text')
    .attr('x', width / 2).attr('y', height + 35)
    .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 12)
    .text('Progress');
  svg.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2).attr('y', -38)
    .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 12)
    .text('Intensity');

  const line = d3.line().x(d => x(d.progress)).y(d => y(d.intensity)).curve(d3.curveMonotoneX);
  const area = d3.area().x(d => x(d.progress)).y0(height).y1(d => y(d.intensity)).curve(d3.curveMonotoneX);

  svg.append('path').datum(env.curve).attr('d', area)
    .attr('fill', '#58a6ff').attr('fill-opacity', 0.1);
  svg.append('path').datum(env.curve).attr('d', line)
    .attr('fill', 'none').attr('stroke', '#58a6ff').attr('stroke-width', 2);

  env.tick_markers.forEach(tm => {
    svg.append('circle')
      .attr('cx', x(tm.progress)).attr('cy', y(tm.intensity)).attr('r', 4)
      .attr('fill', '#58a6ff').attr('stroke', '#0d1117').attr('stroke-width', 1.5);
    svg.append('text')
      .attr('x', x(tm.progress)).attr('y', y(tm.intensity) - 10)
      .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 9)
      .text('t' + tm.tick);
  });
})();

// ---------------------------------------------------------------------------
// 6. Conversation Log (DOM-built, safe)
// ---------------------------------------------------------------------------
(function() {
  const container = document.getElementById('convo-log');
  if (DATA.dialogs.length === 0) {
    container.appendChild(el('div', {style: 'color:var(--text-dim);padding:20px', text: 'No conversations recorded.'}));
    return;
  }

  const byTick = {};
  DATA.dialogs.forEach(d => {
    if (!byTick[d.tick]) byTick[d.tick] = [];
    byTick[d.tick].push(d);
  });

  Object.keys(byTick).sort((a, b) => a - b).forEach(tick => {
    container.appendChild(el('div', {style: 'font-size:13px;font-weight:600;margin:16px 0 8px;color:var(--text-dim)', text: 'Tick ' + tick}));

    byTick[tick].forEach(dialog => {
      const section = el('div', {className: 'convo-section'});

      // Header
      const header = el('div', {className: 'convo-header'});
      header.appendChild(el('span', {className: 'chevron', text: '\u25B6'}));
      header.appendChild(el('span', {text: dialog.participants.join(' \u2194 ')}));
      header.appendChild(el('span', {className: 'pattern-badge', text: dialog.pattern}));
      const turnCount = el('span', {style: 'color:var(--text-dim);font-size:12px;margin-left:auto', text: dialog.turns.length + ' turns'});
      header.appendChild(turnCount);

      // Body
      const body = el('div', {className: 'convo-body'});
      dialog.turns.forEach(turn => {
        const turnDiv = el('div', {className: 'turn'});
        const speaker = el('span', {className: 'speaker', style: 'color:' + nameToColor(turn.speaker_name), text: turn.speaker_name + ': '});
        turnDiv.appendChild(speaker);
        turnDiv.appendChild(document.createTextNode(turn.content));

        if (turn.llm_params) {
          const paramsDiv = el('div', {className: 'params'});
          const p = turn.llm_params;
          if (p.temperature !== undefined)
            paramsDiv.appendChild(el('span', {className: 'param-pill', text: 'temp: ' + p.temperature.toFixed(2)}));
          if (p.top_p !== undefined)
            paramsDiv.appendChild(el('span', {className: 'param-pill', text: 'top_p: ' + p.top_p.toFixed(2)}));
          if (p.max_tokens !== undefined)
            paramsDiv.appendChild(el('span', {className: 'param-pill', text: 'max: ' + p.max_tokens}));
          turnDiv.appendChild(paramsDiv);
        }
        body.appendChild(turnDiv);
      });

      header.addEventListener('click', () => {
        header.classList.toggle('open');
        body.classList.toggle('open');
      });

      section.appendChild(header);
      section.appendChild(body);
      container.appendChild(section);
    });
  });
})();
</script>
</body>
</html>"""
