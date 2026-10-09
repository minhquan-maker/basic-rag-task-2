from dataclasses import dataclass

from .config import Settings
from .models import Chunk, Hit
from .prompts import ANSWER_SYSTEM, VECTOR_PROMPT
from .store import Item, SyncReport, VectorStore

COLLECTION = "chunks"


@dataclass
class VectorAnswer:
    text: str
    hits: list[Hit]

    @property
    def sources(self) -> set[str]:
        return {h.source for h in self.hits}


class VectorRAG:
    """chunk -> embedding -> ChromaDB -> top-k theo cosine -> LLM."""

    def __init__(self, llm, embedder, settings: Settings, store: VectorStore | None = None):
        self.llm, self.embedder, self.s = llm, embedder, settings
        self.store = store or VectorStore.open(settings.chroma_dir, COLLECTION, settings.embed_salt)

    def build(self, chunks: list[Chunk]) -> SyncReport:
        items = [Item(c.id, c.embed_text, c.text, {"source": c.source}) for c in chunks]
        return self.store.sync(items, lambda texts: self.embedder.embed(texts, "document"))

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        vec = self.embedder.embed([question], "query")[0]
        return [Hit(m.id, m.meta.get("source", ""), m.document, m.score)
                for m in self.store.query(vec, k or self.s.top_k)]

    def answer(self, question: str, generate: bool = True) -> VectorAnswer:
        hits = self.retrieve(question)
        if not generate:
            return VectorAnswer("", hits)
        passages = "\n\n".join(f"[{h.id}] {h.text}" for h in hits)
        prompt = VECTOR_PROMPT.format(passages=passages, question=question)
        return VectorAnswer(self.llm.generate(prompt, system=ANSWER_SYSTEM), hits)
