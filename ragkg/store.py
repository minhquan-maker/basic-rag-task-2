import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chromadb


@dataclass(frozen=True)
class Item:
    id: str
    embed_text: str  # văn bản được embedding
    document: str  # văn bản lưu kèm để trả về khi truy hồi
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    id: str
    document: str
    meta: dict
    score: float  # cosine similarity


@dataclass(frozen=True)
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0

    def __str__(self) -> str:
        return (f"+{self.added} mới, ~{self.updated} đổi, -{self.removed} xóa, "
                f"{self.unchanged} giữ nguyên")


class VectorStore:
    """Bọc một collection của ChromaDB (HNSW, cosine). Vector do ta tự tính nên tắt
    embedding_function mặc định. `sync` chỉ embed những mục mới hoặc đã đổi nội dung."""

    def __init__(self, client, name: str, salt: str = ""):
        self.col = client.get_or_create_collection(
            name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        self.salt = salt

    @classmethod
    def open(cls, path: Path, name: str, salt: str = "") -> "VectorStore":
        return cls(chromadb.PersistentClient(path=str(path)), name, salt)

    def _hash(self, item: Item) -> str:
        return hashlib.sha1(f"{self.salt}|{item.embed_text}|{item.document}".encode()).hexdigest()

    def sync(self, items: list[Item], embed: Callable[[list[str]], list[list[float]]]) -> SyncReport:
        got = self.col.get(include=["metadatas"])
        old = {i: (m or {}).get("hash") for i, m in zip(got["ids"], got["metadatas"])}
        wanted = {it.id for it in items}
        stale = [i for i in old if i not in wanted]
        if stale:
            self.col.delete(ids=stale)
        todo = [it for it in items if old.get(it.id) != self._hash(it)]
        if todo:
            self.col.upsert(
                ids=[it.id for it in todo],
                documents=[it.document for it in todo],
                embeddings=embed([it.embed_text for it in todo]),
                metadatas=[{**it.meta, "hash": self._hash(it)} for it in todo],
            )
        updated = sum(1 for it in todo if it.id in old)
        return SyncReport(len(todo) - updated, updated, len(stale), len(items) - len(todo))

    def query(self, vector: list[float], k: int) -> list[Match]:
        n = self.col.count()
        if not n:
            return []
        r = self.col.query(query_embeddings=[vector], n_results=min(k, n),
                           include=["documents", "metadatas", "distances"])
        return [Match(i, d, m or {}, round(1 - dist, 4))
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]
