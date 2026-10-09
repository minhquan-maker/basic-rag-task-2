import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file>#<số thứ tự>"
    source: str  # tên file
    title: str
    text: str

    @property
    def embed_text(self) -> str:
        # Gắn tiêu đề để đoạn chỉ nói "Ông ..." vẫn mang tên chủ thể khi embedding
        return f"{self.title}\n{self.text}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.embed_text.encode()).hexdigest()


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    text: str
    score: float


# Các schema dưới đây được đưa thẳng cho Gemini làm response_schema (structured output)
class Triple(BaseModel):
    subject: str = Field(description="Tên đầy đủ của thực thể chủ thể")
    subject_type: str = Field(description="Loại: Người, Tổ chức, Địa điểm, Giải thưởng, Sự kiện, Công trình, Khái niệm")
    predicate: str = Field(description="Quan hệ ngắn, chữ thường, nối bằng gạch dưới")
    object: str = Field(description="Tên đầy đủ của thực thể đối tượng")
    object_type: str = Field(description="Loại của đối tượng, cùng danh sách với subject_type")


class Extraction(BaseModel):
    triples: list[Triple]


class Mentions(BaseModel):
    entities: list[str]
