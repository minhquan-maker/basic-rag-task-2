import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    chat_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    request_delay: float = float(os.getenv("REQUEST_DELAY", "4"))

    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT / "data"))
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))

    # Đơn vị: ký tự
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    top_k: int = int(os.getenv("TOP_K", "3"))
    kg_hops: int = int(os.getenv("KG_HOPS", "2"))
    kg_max_triples: int = int(os.getenv("KG_MAX_TRIPLES", "30"))
    kg_with_chunks: bool = _bool("KG_WITH_CHUNKS", "true")
    # Ngưỡng cosine để coi một đỉnh trong KG là "cùng thực thể" với cụm từ trong câu hỏi
    entity_match_threshold: float = float(os.getenv("ENTITY_MATCH_THRESHOLD", "0.75"))

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def kg_path(self) -> Path:
        return self.storage_dir / "kg.json"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.json"
