import re
from pathlib import Path

from .models import Chunk

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def load_documents(data_dir: Path) -> list[dict]:
    """Mỗi file .txt/.md là một tài liệu. Dòng đầu dạng `# Tiêu đề` (nếu có) làm tiêu đề."""
    docs = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem.replace("_", " ")
        if text.startswith("# "):
            head, _, text = text.partition("\n")
            title, text = head[2:].strip(), text.strip()
        if text:
            docs.append({"source": path.name, "title": title, "text": text})
    return docs


def _sentences(text: str, limit: int) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        for sent in SENTENCE_END.split(" ".join(para.split())):
            # Câu dài hơn giới hạn thì cắt cứng để mọi đoạn đều không vượt chunk_size
            out.extend(sent[i:i + limit] for i in range(0, len(sent), limit))
    return [s for s in out if s]


def _joined(parts: list[str]) -> int:
    return sum(map(len, parts)) + max(len(parts) - 1, 0)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Gom câu vào đoạn <= size ký tự. Khi sang đoạn mới, các câu cuối của đoạn trước
    (tối đa `overlap` ký tự) được lặp lại để ý không bị đứt ở ranh giới."""
    chunks: list[str] = []
    cur: list[str] = []
    for sent in _sentences(text, size):
        if cur and _joined(cur + [sent]) > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            for s in reversed(cur):
                if _joined([s] + tail) > overlap:
                    break
                tail.insert(0, s)
            cur = tail if _joined(tail + [sent]) <= size else []
        cur.append(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_documents(docs: list[dict], size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(f"{d['source']}#{i}", d["source"], d["title"], piece)
        for d in docs
        for i, piece in enumerate(chunk_text(d["text"], size, overlap))
    ]
