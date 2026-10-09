import json
from dataclasses import dataclass
from pathlib import Path

from .graph import fold


@dataclass
class Case:
    id: str
    question: str
    gold_sources: list[str]  # các file phải được truy hồi để trả lời được
    answer_contains: list[str]  # các cụm phải có trong câu trả lời đúng
    hops: int  # số tài liệu cần ghép


def load_cases(path: Path) -> list[Case]:
    return [Case(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def source_recall(gold: list[str], retrieved: set[str]) -> float:
    return sum(g in retrieved for g in gold) / len(gold) if gold else 1.0


def answer_ok(answer: str, needles: list[str]) -> bool:
    text = fold(answer)
    return all(fold(n) in text for n in needles)


def run(cases: list[Case], vector, graph, generate: bool = True) -> list[dict]:
    rows = []
    for c in cases:
        row = {"id": c.id, "hops": c.hops, "question": c.question}
        for name, pipe in (("rag", vector), ("kg", graph)):
            ans = pipe.answer(c.question, generate=generate)
            row[name] = {
                "recall": source_recall(c.gold_sources, ans.sources),
                "ok": answer_ok(ans.text, c.answer_contains) if generate else None,
                "answer": ans.text,
            }
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> str:
    def mean(sel, key, name):
        vals = [r[name][key] for r in sel if r[name][key] is not None]
        return f"{sum(vals) / len(vals):.2f}" if vals else "  - "

    lines = [f"{'số tài liệu cần ghép':<22}{'n':>3}  {'recall RAG':>10} {'recall KG':>10}  {'đúng RAG':>9} {'đúng KG':>8}"]
    groups = [(f"{h} tài liệu", [r for r in rows if r["hops"] == h]) for h in sorted({r["hops"] for r in rows})]
    for label, sel in groups + [("tất cả", rows)]:
        lines.append(f"{label:<22}{len(sel):>3}  {mean(sel, 'recall', 'rag'):>10} {mean(sel, 'recall', 'kg'):>10}  "
                     f"{mean(sel, 'ok', 'rag'):>9} {mean(sel, 'ok', 'kg'):>8}")
    return "\n".join(lines)
