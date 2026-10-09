import hashlib
import json
import logging
import time
from pathlib import Path

from .graph import KnowledgeGraph
from .models import Chunk, Extraction
from .prompts import EXTRACT_PROMPT, EXTRACT_SYSTEM

log = logging.getLogger(__name__)

# Tăng khi đổi prompt/schema để vô hiệu hóa cache cũ
PROMPT_VERSION = "1"


class Extractor:
    """Trích bộ ba từ từng đoạn bằng LLM, có cache theo nội dung đoạn + model + phiên bản prompt.
    Index lại sau khi sửa một file chỉ gọi LLM cho những đoạn thật sự đổi."""

    def __init__(self, llm, cache_path: Path, model_tag: str, interval: float = 0.0):
        self.llm, self.path, self.tag, self.interval = llm, cache_path, model_tag, interval
        self.cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.calls = self.hits = 0

    def _key(self, chunk: Chunk) -> str:
        raw = f"{PROMPT_VERSION}|{self.tag}|{chunk.content_hash}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def extract(self, chunk: Chunk) -> Extraction:
        key = self._key(chunk)
        if key in self.cache:
            self.hits += 1
            return Extraction.model_validate(self.cache[key])
        prompt = EXTRACT_PROMPT.format(title=chunk.title, text=chunk.text)
        result = self.llm.generate_structured(prompt, Extraction, system=EXTRACT_SYSTEM)
        self.calls += 1
        self.cache[key] = result.model_dump()
        self._save()  # lưu ngay để lỡ dừng giữa chừng vẫn giữ phần đã trích
        if self.interval:
            time.sleep(self.interval)
        return result


def build_graph(chunks: list[Chunk], extractor: Extractor) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for n, chunk in enumerate(chunks, 1):
        try:
            result = extractor.extract(chunk)
        except Exception as e:  # một đoạn lỗi không làm hỏng cả đợt index; lần sau sẽ thử lại
            log.warning("[%d/%d] %s: bỏ qua (%s)", n, len(chunks), chunk.id, e)
            continue
        for t in result.triples:
            kg.add_triple(t.subject, t.predicate, t.object, chunk.id, t.subject_type, t.object_type)
    return kg
