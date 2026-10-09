import re
from pathlib import Path


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append({"source": path.name, "text": text})
    return docs


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Gom câu thành đoạn <= chunk_size ký tự; các câu cuối của đoạn trước
    (tổng <= overlap ký tự) được lặp lại ở đầu đoạn sau để không đứt mạch ý."""
    chunks, current = [], []
    for sent in split_sentences(text):
        if current and len(" ".join(current + [sent])) > chunk_size:
            chunks.append(" ".join(current))
            tail, length = [], 0
            for s in reversed(current):
                if length + len(s) > overlap:
                    break
                tail.insert(0, s)
                length += len(s)
            current = tail
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunks(docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    return [
        {"id": f"{d['source']}#{i}", "source": d["source"], "text": piece}
        for d in docs
        for i, piece in enumerate(chunk_text(d["text"], chunk_size, overlap))
    ]
