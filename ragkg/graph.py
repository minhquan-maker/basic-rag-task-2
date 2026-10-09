import json
import re
import unicodedata
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


def fold(s: str) -> str:
    """NFC + casefold. NFC quan trọng với tiếng Việt: "ế" dựng sẵn và "ê" + dấu rời là hai chuỗi khác nhau."""
    return unicodedata.normalize("NFC", str(s)).casefold()


def normalize(name: str) -> str:
    """Khóa định danh của thực thể: không phân biệt hoa/thường, gọn khoảng trắng, bỏ dấu câu hai đầu."""
    return re.sub(r"\s+", " ", fold(name)).strip(" .,;:!?\"'()[]")


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(name))).strip()


def snake(relation: str) -> str:
    return re.sub(r"[\s\-]+", "_", fold(relation).strip()).strip("_")


class KnowledgeGraph:
    """Đồ thị có hướng, nhiều cạnh (networkx.MultiDiGraph).

    Đỉnh: khóa = tên chuẩn hóa; thuộc tính name, type, aliases, sources (id các đoạn nhắc tới).
    Cạnh: khóa = tên quan hệ, nên cùng (chủ thể, quan hệ, đối tượng) chỉ có một cạnh;
    thuộc tính sources lưu các đoạn là bằng chứng.
    """

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()
        self._alias: dict[str, str] = {}
        for key, data in self.g.nodes(data=True):
            self._index(key, data)

    def _index(self, key: str, data: dict):
        self._alias[key] = key
        for a in data["aliases"]:
            self._alias.setdefault(normalize(a), key)

    def add_entity(self, name: str, etype: str = "", source: str = "") -> str | None:
        key = self.find(name) or normalize(name)
        if not key:
            return None
        if key not in self.g:
            self.g.add_node(key, name=clean(name), type=etype, aliases=[], sources=[])
            self._index(key, self.g.nodes[key])
        node = self.g.nodes[key]
        node["type"] = node["type"] or etype
        if source and source not in node["sources"]:
            node["sources"].append(source)
        return key

    def add_triple(self, subject: str, predicate: str, obj: str, source: str = "",
                   subject_type: str = "", object_type: str = ""):
        s = self.add_entity(subject, subject_type, source)
        o = self.add_entity(obj, object_type, source)
        rel = snake(predicate)
        if not (s and o and rel) or s == o:
            return
        if self.g.has_edge(s, o, key=rel):
            srcs = self.g.edges[s, o, rel]["sources"]
            if source and source not in srcs:
                srcs.append(source)
        else:
            self.g.add_edge(s, o, key=rel, relation=rel, sources=[source] if source else [])

    def find(self, mention: str) -> str | None:
        """Khóa của đỉnh có tên hoặc bí danh khớp chính xác (sau chuẩn hóa)."""
        return self._alias.get(normalize(mention))

    def surfaces(self) -> list[tuple[str, str]]:
        """(cách viết đã chuẩn hóa, khóa đỉnh) của mọi tên và bí danh."""
        return list(self._alias.items())

    def name(self, key: str) -> str:
        return self.g.nodes[key]["name"]

    def merge(self, alias_key: str, canon_key: str):
        """Gộp đỉnh `alias_key` vào `canon_key`: chuyển hết cạnh, nguồn và giữ tên cũ làm bí danh."""
        a, c = self.g.nodes[alias_key], self.g.nodes[canon_key]
        for u, v, rel, d in list(self.g.edges(alias_key, keys=True, data=True)) + \
                list(self.g.in_edges(alias_key, keys=True, data=True)):
            u2, v2 = (canon_key if u == alias_key else u), (canon_key if v == alias_key else v)
            if u2 == v2:
                continue
            if self.g.has_edge(u2, v2, key=rel):
                srcs = self.g.edges[u2, v2, rel]["sources"]
                srcs.extend(s for s in d["sources"] if s not in srcs)
            else:
                self.g.add_edge(u2, v2, key=rel, relation=rel, sources=list(d["sources"]))
        for label in [a["name"], *a["aliases"]]:
            if normalize(label) != canon_key and label not in c["aliases"]:
                c["aliases"].append(label)
        c["type"] = c["type"] or a["type"]
        c["sources"].extend(s for s in a["sources"] if s not in c["sources"])
        self.g.remove_node(alias_key)
        for surface, key in list(self._alias.items()):
            if key == alias_key:
                self._alias[surface] = canon_key

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
