from pathlib import Path

from pyvis.network import Network

from .kg_builder import KnowledgeGraph


def export_html(kg: KnowledgeGraph, out_path: Path) -> Path:
    net = Network(height="750px", width="100%", directed=True, cdn_resources="remote")
    for key, data in kg.g.nodes(data=True):
        net.add_node(key, label=data["name"], title=data.get("type") or "",
                     size=10 + 3 * kg.g.degree(key))
    for h, t, data in kg.g.edges(data=True):
        net.add_edge(h, t, label=data["relation"], title=", ".join(data.get("sources", [])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(net.generate_html(), encoding="utf-8")
    return out_path
