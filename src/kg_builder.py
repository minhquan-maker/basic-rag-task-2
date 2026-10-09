import json
import re
import time
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from .prompts import EXTRACT_PROMPT, EXTRACT_SYSTEM


def normalize(name: str) -> str:
    """Khóa gộp thực thể trùng: bỏ khoảng trắng thừa, dấu câu hai đầu, không phân biệt hoa/thường."""
    return re.sub(r"\s+", " ", str(name)).strip(" .,;:\"'()").casefold()


class KnowledgeGraph:
    """MultiDiGraph của NetworkX.
    Đỉnh: key = tên chuẩn hóa; thuộc tính name, type, sources (id các đoạn nhắc tới).
    Cạnh: key = tên quan hệ; thuộc tính relation, sources (bằng chứng)."""

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()

    def add_entity(self, name: str, etype: str = "", source: str = "") -> str | None:
        key = normalize(name)
        if not key:
            return None
        if key not in self.g:
            self.g.add_node(key, name=re.sub(r"\s+", " ", str(name)).strip(), type=etype, sources=[])
        node = self.g.nodes[key]
        node["type"] = node["type"] or etype
        if source and source not in node["sources"]:
            node["sources"].append(source)
        return key

    def add_triple(self, head: str, relation: str, tail: str, source: str = ""):
        h, t = self.add_entity(head, source=source), self.add_entity(tail, source=source)
        rel = re.sub(r"\s+", "_", str(relation).strip().lower())
        if not (h and t and rel) or h == t:
            return
        if self.g.has_edge(h, t, key=rel):
            srcs = self.g.edges[h, t, rel]["sources"]
            if source and source not in srcs:
                srcs.append(source)
        else:
            self.g.add_edge(h, t, key=rel, relation=rel, sources=[source] if source else [])

    def name(self, key: str) -> str:
        return self.g.nodes[key]["name"]

    def stats(self) -> str:
        return f"{self.g.number_of_nodes()} thực thể, {self.g.number_of_edges()} quan hệ"

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.g, edges="edges")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(json_graph.node_link_graph(data, directed=True, multigraph=True, edges="edges"))


def extract_triples(llm, text: str) -> dict:
    data = llm.generate_json(EXTRACT_PROMPT.format(text=text), system=EXTRACT_SYSTEM)
    if not isinstance(data, dict):
        data = {}
    return {"entities": data.get("entities") or [], "relations": data.get("relations") or []}


def build_kg(chunks: list[dict], llm, delay: float = 0.0) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {chunk['id']}")
        try:
            data = extract_triples(llm, chunk["text"])
        except Exception as e:  # một đoạn lỗi không làm hỏng cả quá trình
            print(f"    ! bỏ qua: {e}")
            continue
        for ent in data["entities"]:
            if isinstance(ent, dict) and ent.get("name"):
                kg.add_entity(ent["name"], ent.get("type", ""), chunk["id"])
        for rel in data["relations"]:
            if isinstance(rel, dict) and all(rel.get(k) for k in ("head", "relation", "tail")):
                kg.add_triple(rel["head"], rel["relation"], rel["tail"], chunk["id"])
        if delay and i < len(chunks):
            time.sleep(delay)
    return kg
