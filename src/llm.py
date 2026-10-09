import json
import re
import time

from google import genai
from google.genai import errors, types

from .config import Config


def _retry(fn, max_tries: int = 5):
    """Thử lại với thời gian chờ tăng dần khi gặp 429 (hết hạn mức) hoặc 5xx."""
    for attempt in range(max_tries):
        try:
            return fn()
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < max_tries - 1:
                wait = 5 * 2 ** attempt
                print(f"  [Gemini {e.code}] thử lại sau {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)  # bỏ rào ```json nếu có
    return json.loads(m.group(1) if m else text)


class GeminiLLM:
    """Giao diện LLM tối thiểu: generate / generate_json / embed.
    Test thay lớp này bằng FakeLLM nên pipeline không phụ thuộc trực tiếp vào SDK."""

    def __init__(self, cfg: Config):
        if not cfg.api_key:
            raise RuntimeError("Chưa có GEMINI_API_KEY: copy .env.example thành .env rồi điền key.")
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.api_key)

    def _call(self, prompt, system, temperature, json_mode=False):
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.cfg.chat_model, contents=prompt, config=config))
        return resp.text or ""

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return self._call(prompt, system, temperature)

    def generate_json(self, prompt: str, system: str | None = None):
        return parse_json(self._call(prompt, system, 0.0, json_mode=True) or "{}")

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """task_type: RETRIEVAL_DOCUMENT (đoạn văn), RETRIEVAL_QUERY (câu hỏi),
        SEMANTIC_SIMILARITY (tên thực thể). Gửi theo lô 50 phần tử."""
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.cfg.embed_dim)
        out = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            resp = _retry(lambda: self.client.models.embed_content(
                model=self.cfg.embed_model, contents=batch, config=config))
            out.extend(e.values for e in resp.embeddings)
        return out
