import json
import re
from collections import deque

import chromadb

from .config import Config
from .kg_builder import KnowledgeGraph, build_kg, normalize
from .prompts import ANSWER_SYSTEM, KG_RAG_PROMPT, QUESTION_ENTITY_PROMPT

ENTITY_COLLECTION = "entities"


class KGRAG:
    """KG-RAG cơ bản: chunk -> LLM trích bộ ba -> đồ thị -> liên kết thực thể trong câu hỏi
    -> đồ thị con k-hop -> LLM trả lời."""

    def __init__(self, llm, cfg: Config, client=None):
        self.llm = llm
        self.cfg = cfg
        self.client = client or chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self.kg = KnowledgeGraph.load(cfg.kg_path) if cfg.kg_path.exists() else KnowledgeGraph()
        self.chunks = (json.loads(cfg.chunks_path.read_text(encoding="utf-8"))
                       if cfg.chunks_path.exists() else {})

    def _collection(self):
        return self.client.get_or_create_collection(
            name=ENTITY_COLLECTION, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}})

    def build_index(self, chunks: list[dict]) -> str:
        self.kg = build_kg(chunks, self.llm, delay=self.cfg.request_delay)
        self.chunks = {c["id"]: c["text"] for c in chunks}
        self.kg.save(self.cfg.kg_path)
        self.cfg.chunks_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        # Embedding tên thực thể: để khớp "Einstein" trong câu hỏi với đỉnh "Albert Einstein"
        try:
            self.client.delete_collection(ENTITY_COLLECTION)
        except Exception:
            pass
        keys = list(self.kg.g.nodes)
        if keys:
            names = [self.kg.name(k) for k in keys]
            self._collection().add(
                ids=keys, documents=names,
                embeddings=self.llm.embed(names, "SEMANTIC_SIMILARITY"))
        return self.kg.stats()

    def link_entities(self, question: str) -> list[str]:
        """Tìm các đỉnh gốc: (1) tên đỉnh xuất hiện nguyên văn trong câu hỏi;
        (2) LLM trích cụm thực thể -> khớp khóa chuẩn hóa -> nếu không khớp thì lấy đỉnh
        gần nhất theo embedding khi cosine >= entity_match_threshold."""
        found = []
        q = question.casefold()
        for key in self.kg.g.nodes:
            if len(key) > 2 and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", q):
                found.append(key)
        try:
            mentions = self.llm.generate_json(
                QUESTION_ENTITY_PROMPT.format(question=question)).get("entities", [])
        except Exception:
            mentions = []
        unmatched = []
        for m in mentions:
            key = normalize(m)
            if key in self.kg.g:
                found.append(key)
            elif key:
                unmatched.append(m)
        if unmatched and self.kg.g.number_of_nodes():
            res = self._collection().query(
                query_embeddings=self.llm.embed(unmatched, "SEMANTIC_SIMILARITY"), n_results=1)
            for ids, dists in zip(res["ids"], res["distances"]):
                if ids and 1 - dists[0] >= self.cfg.entity_match_threshold:
                    found.append(ids[0])
        return list(dict.fromkeys(found))

    def subgraph_edges(self, seeds: list[str], hops: int | None = None,
                       max_triples: int | None = None) -> list[tuple[str, str, str, list[str]]]:
        """BFS vô hướng từ các đỉnh gốc tới độ sâu `hops`; trả về (head, relation, tail, sources)
        của các cạnh nằm trong đồ thị con, cạnh gần gốc hơn được ưu tiên khi cắt theo max_triples."""
        hops = self.cfg.kg_hops if hops is None else hops
        max_triples = max_triples or self.cfg.kg_max_triples
        g = self.kg.g
        dist = {s: 0 for s in seeds if s in g}
        queue = deque(dist)
        while queue:
            u = queue.popleft()
            if dist[u] >= hops:
                continue
            for v in list(g.successors(u)) + list(g.predecessors(u)):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        edges = [
            (min(dist[h], dist[t]), h, d["relation"], t, d["sources"])
            for h, t, d in g.edges(data=True)
            if h in dist and t in dist and min(dist[h], dist[t]) < hops
        ]
        edges.sort(key=lambda e: e[0])
        return [(self.kg.name(h), r, self.kg.name(t), s) for _, h, r, t, s in edges[:max_triples]]

    def answer(self, question: str) -> dict:
        seeds = self.link_entities(question)
        edges = self.subgraph_edges(seeds)
        if not edges:
            return {"answer": "Không tìm thấy thực thể liên quan trong đồ thị tri thức.",
                    "entities": [], "triples": []}
        triples = "\n".join(f"{h} --[{r}]--> {t}" for h, r, t, _ in edges)
        chunks = ""
        if self.cfg.kg_with_chunks:
            ids = list(dict.fromkeys(s for *_, srcs in edges for s in srcs))
            body = "\n".join(f"[{i}] {self.chunks[i]}" for i in ids if i in self.chunks)
            if body:
                chunks = f"\nĐOẠN VĂN GỐC CỦA CÁC SỰ KIỆN TRÊN:\n{body}\n"
        prompt = KG_RAG_PROMPT.format(triples=triples, chunks=chunks, question=question)
        return {"answer": self.llm.generate(prompt, system=ANSWER_SYSTEM),
                "entities": [self.kg.name(s) for s in seeds],
                "triples": [(h, r, t) for h, r, t, _ in edges]}
