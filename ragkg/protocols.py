from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
EmbedKind = Literal["document", "query", "similarity"]


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def generate_structured(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...


class Embedder(Protocol):
    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]: ...
