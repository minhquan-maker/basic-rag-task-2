import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field

from .config import Settings
from .extract import Extractor, build_graph
from .graph import KnowledgeGraph, fold
from .models import Chunk, Mentions
from .prompts import ANSWER_SYSTEM, GRAPH_PROMPT, MENTION_PROMPT
from .resolve import merge_aliases
from .store import Item, SyncReport, VectorStore

log = logging.getLogger(__name__)
COLLECTION = "entities"


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    sources: tuple[str, ...]  # id các đoạn là bằng chứng
    distance: int  # số bước từ đỉnh gốc gần nhất

    def __str__(self) -> str:
        return f"{self.subject} --[{self.predicate}]--> {self.object}"


@dataclass
class GraphContext:
    seeds: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    passages: list[tuple[str, str]] = field(default_factory=list)  # (id đoạn, nội dung)

    @property
    def sources(self) -> set[str]:
        """Tên các file có đoạn đóng vai bằng chứng cho các sự kiện đã lấy."""
        return {c.split("#")[0] for f in self.facts for c in f.sources}


@dataclass
class GraphAnswer:
    text: str
    context: GraphContext

    @property
    def sources(self) -> set[str]:
        return self.context.sources


class GraphRAG:
    """chunk -> LLM trích bộ ba -> KnowledgeGraph -> liên kết thực thể trong câu hỏi
    -> đồ thị con k-hop (kèm đoạn văn gốc) -> LLM."""

    def __init__(self, llm, embedder, settings: Settings, store: VectorStore | None = None):
        self.llm, self.embedder, self.s = llm, embedder, settings
        self.store = store or VectorStore.open(settings.chroma_dir, COLLECTION, settings.embed_salt)
        self.kg = KnowledgeGraph.load(settings.kg_path) if settings.kg_path.exists() else KnowledgeGraph()
        self.chunks: dict[str, str] = (
            json.loads(settings.chunks_path.read_text(encoding="utf-8"))
            if settings.chunks_path.exists() else {})

    # ---- index ----
    def build(self, chunks: list[Chunk], fresh: bool = False) -> tuple[KnowledgeGraph, SyncReport]:
        if fresh and self.s.extract_cache_path.exists():
            self.s.extract_cache_path.unlink()
        extractor = Extractor(self.llm, self.s.extract_cache_path,
                              f"{self.s.chat_model}", self.s.request_interval)
        self.kg = build_graph(chunks, extractor)
        log.info("Trích bộ ba: %d lần gọi LLM, %d đoạn lấy từ cache", extractor.calls, extractor.hits)
        if self.s.alias_merge:
            for alias, canon in merge_aliases(self.kg):
                log.info("Gộp thực thể: %r -> %r", alias, canon)
        self.chunks = {c.id: c.text for c in chunks}
        self.kg.save(self.s.kg_path)
        self.s.chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        items = [Item(k, self.kg.name(k), self.kg.name(k)) for k in self.kg.g.nodes]
        report = self.store.sync(items, lambda texts: self.embedder.embed(texts, "similarity"))
        return self.kg, report

    # ---- truy hồi ----
    def link(self, question: str) -> list[str]:
        """Tìm các đỉnh gốc từ câu hỏi, theo thứ tự tin cậy giảm dần:
        1. tên/bí danh của đỉnh xuất hiện nguyên văn (ưu tiên cụm dài, tránh khớp "Turing" bên trong "Giải Turing");
        2. LLM tách cụm thực thể trong câu hỏi, khớp chính xác với tên/bí danh;
        3. cụm còn lại: đỉnh gần nhất theo embedding nếu cosine >= entity_threshold."""
        found: list[str] = []
        text = fold(question)
        taken: list[tuple[int, int]] = []
        for surface, key in sorted(self.kg.surfaces(), key=lambda p: -len(p[0])):
            if len(surface) < 2:
                continue
            for m in re.finditer(rf"(?<!\w){re.escape(surface)}(?!\w)", text):
                if not any(m.start() < e and s < m.end() for s, e in taken):
                    taken.append(m.span())
                    found.append(key)
        unmatched = []
        try:
            mentions = self.llm.generate_structured(
                MENTION_PROMPT.format(question=question), Mentions).entities
        except Exception as e:
            log.warning("Không tách được thực thể bằng LLM: %s", e)
            mentions = []
        for m in mentions:
            key = self.kg.find(m)
            if key:
                found.append(key)
            elif m.strip():
                unmatched.append(m)
        if unmatched and self.kg.g.number_of_nodes():
            for vec in self.embedder.embed(unmatched, "similarity"):
                best = self.store.query(vec, 1)
                if best and best[0].score >= self.s.entity_threshold:
                    found.append(best[0].id)
        return list(dict.fromkeys(found))

    def expand(self, seeds: list[str]) -> list[Fact]:
        """BFS vô hướng từ các đỉnh gốc tới `kg_hops` bước. Đỉnh có hơn `kg_hub_limit` cạnh
        (không phải đỉnh gốc) không được mở rộng tiếp. Cạnh gần gốc và nhiều bằng chứng đi trước."""
        g = self.kg.g
        dist = {s: 0 for s in seeds if s in g}
        queue = deque(dist)
        while queue:
            u = queue.popleft()
            if dist[u] >= self.s.kg_hops or (dist[u] > 0 and g.degree(u) > self.s.kg_hub_limit):
                continue
            for v in (*g.successors(u), *g.predecessors(u)):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        facts = [
            Fact(self.kg.name(u), d["relation"], self.kg.name(v), tuple(d["sources"]), min(dist[u], dist[v]))
            for u, v, d in g.edges(data=True)
            if u in dist and v in dist and min(dist[u], dist[v]) < self.s.kg_hops
        ]
        facts.sort(key=lambda f: (f.distance, -len(f.sources), f.subject, f.predicate))
        return facts[:self.s.kg_max_facts]

    def retrieve(self, question: str) -> GraphContext:
        seeds = self.link(question)
        facts = self.expand(seeds)
        votes: dict[str, int] = {}
        for f in facts:
            for c in f.sources:
                votes[c] = votes.get(c, 0) + 1
        ranked = sorted(votes, key=lambda c: -votes[c])  # sort ổn định: hòa thì giữ thứ tự xuất hiện
        passages = [(c, self.chunks[c]) for c in ranked if c in self.chunks][:self.s.kg_max_passages]
        return GraphContext([self.kg.name(s) for s in seeds], facts, passages)

    def answer(self, question: str, generate: bool = True) -> GraphAnswer:
        ctx = self.retrieve(question)
        if not ctx.facts:
            return GraphAnswer("Không tìm thấy thực thể nào của câu hỏi trong đồ thị tri thức.", ctx)
        if not generate:
            return GraphAnswer("", ctx)
        prompt = GRAPH_PROMPT.format(
            facts="\n".join(map(str, ctx.facts)),
            passages="\n\n".join(f"[{c}] {t}" for c, t in ctx.passages) or "(không có)",
            question=question)
        return GraphAnswer(self.llm.generate(prompt, system=ANSWER_SYSTEM), ctx)
