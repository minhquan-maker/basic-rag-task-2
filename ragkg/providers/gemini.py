import logging
import time

from google import genai
from google.genai import errors, types

from ..config import Settings
from ..protocols import EmbedKind, T

log = logging.getLogger(__name__)

TASK_TYPE = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
    "similarity": "SEMANTIC_SIMILARITY",
}
RETRYABLE = (429, 500, 503, 504)
EMBED_BATCH = 50


class GeminiProvider:
    """Cài đặt cả LLM lẫn Embedder trên Gemini API."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError("Thiếu GEMINI_API_KEY: copy .env.example thành .env rồi điền key.")
        self.s = settings
        self.client = genai.Client(api_key=settings.api_key)

    @staticmethod
    def _retry(fn, tries: int = 5):
        for attempt in range(tries):
            try:
                return fn()
            except errors.APIError as e:
                if e.code not in RETRYABLE or attempt == tries - 1:
                    raise
                wait = 5 * 2**attempt
                log.warning("Gemini trả %s, thử lại sau %ss", e.code, wait)
                time.sleep(wait)

    def _generate(self, prompt: str, config: types.GenerateContentConfig):
        return self._retry(lambda: self.client.models.generate_content(
            model=self.s.chat_model, contents=prompt, config=config))

    def generate(self, prompt: str, system: str | None = None) -> str:
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.2)
        return self._generate(prompt, config).text or ""

    def generate_structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.0,
            response_mime_type="application/json", response_schema=schema)
        resp = self._generate(prompt, config)
        if isinstance(resp.parsed, schema):
            return resp.parsed
        return schema.model_validate_json(resp.text or "{}")

    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=TASK_TYPE[kind], output_dimensionality=self.s.embed_dim)
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            resp = self._retry(lambda: self.client.models.embed_content(
                model=self.s.embed_model, contents=batch, config=config))
            out.extend(e.values for e in resp.embeddings)
        return out
