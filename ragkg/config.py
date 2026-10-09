import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    request_interval: float = 4.0

    data_dir: Path = ROOT / "data"
    storage_dir: Path = ROOT / "storage"

    chunk_size: int = 600  # ký tự
    chunk_overlap: int = 100
    top_k: int = 3

    kg_hops: int = 2
    kg_max_facts: int = 30
    kg_max_passages: int = 4
    kg_hub_limit: int = 25
    entity_threshold: float = 0.75
    alias_merge: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.getenv
        return cls(
            api_key=e("GEMINI_API_KEY", ""),
            chat_model=e("GEMINI_MODEL", cls.chat_model),
            embed_model=e("GEMINI_EMBED_MODEL", cls.embed_model),
            embed_dim=int(e("EMBED_DIM", cls.embed_dim)),
            request_interval=float(e("REQUEST_INTERVAL", cls.request_interval)),
            data_dir=Path(e("DATA_DIR", cls.data_dir)),
            storage_dir=Path(e("STORAGE_DIR", cls.storage_dir)),
            chunk_size=int(e("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(e("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(e("TOP_K", cls.top_k)),
            kg_hops=int(e("KG_HOPS", cls.kg_hops)),
            kg_max_facts=int(e("KG_MAX_FACTS", cls.kg_max_facts)),
            kg_max_passages=int(e("KG_MAX_PASSAGES", cls.kg_max_passages)),
            kg_hub_limit=int(e("KG_HUB_LIMIT", cls.kg_hub_limit)),
            entity_threshold=float(e("ENTITY_THRESHOLD", cls.entity_threshold)),
            alias_merge=_flag("ALIAS_MERGE", "true"),
        )

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def kg_path(self) -> Path:
        return self.storage_dir / "kg.json"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.json"

    @property
    def extract_cache_path(self) -> Path:
        return self.storage_dir / "extract_cache.json"

    @property
    def embed_salt(self) -> str:
        """Đổi model hoặc số chiều thì vector cũ không còn dùng được, phải embed lại."""
        return f"{self.embed_model}:{self.embed_dim}"
