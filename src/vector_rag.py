import chromadb

from .config import Config
from .prompts import ANSWER_SYSTEM, VECTOR_RAG_PROMPT

COLLECTION = "chunks"


class VectorRAG:
    """RAG cơ bản: chunk -> embedding -> ChromaDB -> top-k -> LLM."""

    def __init__(self, llm, cfg: Config, client=None):
        self.llm = llm
        self.cfg = cfg
        self.client = client or chromadb.PersistentClient(path=str(cfg.chroma_dir))

    def _collection(self):
        # embedding_function=None: vector do ta tự tính bằng Gemini; khoảng cách cosine
        return self.client.get_or_create_collection(
            name=COLLECTION, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}})

    def build_index(self, chunks: list[dict]) -> int:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = self._collection()
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=self.llm.embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT"),
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        return col.count()

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        q_vec = self.llm.embed([question], "RETRIEVAL_QUERY")[0]
        res = self._collection().query(query_embeddings=[q_vec], n_results=k or self.cfg.top_k)
        # cosine similarity = 1 - cosine distance
        return [
            {"id": i, "text": t, "score": round(1 - d, 4)}
            for i, t, d in zip(res["ids"][0], res["documents"][0], res["distances"][0])
        ]

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = VECTOR_RAG_PROMPT.format(context=context, question=question)
        return {"answer": self.llm.generate(prompt, system=ANSWER_SYSTEM), "contexts": hits}
