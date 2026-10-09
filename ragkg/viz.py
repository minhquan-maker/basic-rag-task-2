from pathlib import Path

from pyvis.network import Network

from .graph import KnowledgeGraph

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]


def export_html(kg: KnowledgeGraph, out_path: Path) -> Path:
    net = Network(height="800px", width="100%", directed=True, cdn_resources="remote")
    colors: dict[str, str] = {}
    for key, d in kg.g.nodes(data=True):
        color = colors.setdefault(d["type"] or "?", PALETTE[len(colors) % len(PALETTE)])
        aliases = f" (còn gọi: {', '.join(d['aliases'])})" if d["aliases"] else ""
        net.add_node(key, label=d["name"], color=color, size=10 + 3 * kg.g.degree(key),
                     title=f"{d['type'] or '?'}{aliases}")
    for u, v, d in kg.g.edges(data=True):
        net.add_edge(u, v, label=d["relation"], title=", ".join(d["sources"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(net.generate_html(), encoding="utf-8")
    return out_path
