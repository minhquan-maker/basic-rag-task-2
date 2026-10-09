import argparse
import json
import logging

from .config import Settings
from .corpus import chunk_documents, load_documents
from .graph import KnowledgeGraph


def providers(s: Settings):
    from .providers.gemini import GeminiProvider
    p = GeminiProvider(s)
    return p, p  # cùng một đối tượng đóng cả hai vai LLM và Embedder


def load_chunks(s: Settings):
    docs = load_documents(s.data_dir)
    chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    print(f"{len(docs)} tài liệu -> {len(chunks)} đoạn")
    return chunks


def cmd_index(s: Settings, a):
    from .graph_rag import GraphRAG
    from .vector_rag import VectorRAG
    llm, emb = providers(s)
    chunks = load_chunks(s)
    if a.only in (None, "rag"):
        print("[RAG]", VectorRAG(llm, emb, s).build(chunks))
    if a.only in (None, "kg"):
        kg, report = GraphRAG(llm, emb, s).build(chunks, fresh=a.fresh)
        print(f"[KG] {kg.stats()}; embedding thực thể: {report}")


def print_rag(ans):
    print("\n== RAG thường ==")
    for h in ans.hits:
        print(f"  [{h.id}] cos={h.score}  {h.text[:70]}...")
    if ans.text:
        print("Trả lời:", ans.text.strip())


def print_kg(ans):
    ctx = ans.context
    print("\n== KG-RAG ==")
    print("  Đỉnh gốc:", ", ".join(ctx.seeds) or "(không có)")
    for f in ctx.facts:
        print(f"  d={f.distance} {f}   <- {', '.join(f.sources)}")
    if ans.text:
        print("Trả lời:", ans.text.strip())


def ask(s: Settings, llm, emb, question: str, mode: str, generate: bool = True):
    from .graph_rag import GraphRAG
    from .vector_rag import VectorRAG
    if mode in ("rag", "both"):
        print_rag(VectorRAG(llm, emb, s).answer(question, generate))
    if mode in ("kg", "both"):
        print_kg(GraphRAG(llm, emb, s).answer(question, generate))


def cmd_ask(s: Settings, a):
    ask(s, *providers(s), a.question, a.mode, generate=not a.no_answer)


def cmd_chat(s: Settings, a):
    llm, emb = providers(s)
    print("Nhập câu hỏi, Enter trống để thoát.")
    while q := input("\n> ").strip():
        ask(s, llm, emb, q, a.mode)


def cmd_graph(s: Settings, a):
    from .viz import export_html
    if not s.kg_path.exists():
        return print("Chưa có đồ thị. Chạy `ragkg index` trước.")
    kg = KnowledgeGraph.load(s.kg_path)
    print(kg.stats())
    for u, v, d in sorted(kg.g.edges(data=True), key=lambda e: e[2]["relation"]):
        print(f"  {kg.name(u)} --[{d['relation']}]--> {kg.name(v)}")
    print("Đã xuất", export_html(kg, s.storage_dir / "kg.html"))


def cmd_eval(s: Settings, a):
    from . import evaluate
    from .graph_rag import GraphRAG
    from .vector_rag import VectorRAG
    llm, emb = providers(s)
    cases = evaluate.load_cases(a.cases)
    rows = evaluate.run(cases, VectorRAG(llm, emb, s), GraphRAG(llm, emb, s), generate=not a.no_answer)
    for r in rows:
        flag = lambda x: "-" if x["ok"] is None else ("đúng" if x["ok"] else "sai")  # noqa: E731
        print(f"{r['id']:<4} {r['hops']}  recall RAG={r['rag']['recall']:.2f} KG={r['kg']['recall']:.2f}  "
              f"RAG:{flag(r['rag'])} KG:{flag(r['kg'])}  {r['question']}")
    print("\n" + evaluate.summarize(rows))
    out = s.storage_dir / "eval.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Chi tiết:", out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragkg", description="Vector RAG và KG-RAG cơ bản trên Gemini")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("index", help="xây (hoặc cập nhật) vector index và knowledge graph")
    q.add_argument("--only", choices=["rag", "kg"])
    q.add_argument("--fresh", action="store_true", help="bỏ cache trích bộ ba, gọi lại LLM cho mọi đoạn")
    q.set_defaults(func=cmd_index)

    for name, fn in (("ask", cmd_ask), ("chat", cmd_chat)):
        q = sub.add_parser(name, help="hỏi một câu" if name == "ask" else "hỏi đáp liên tục")
        if name == "ask":
            q.add_argument("question")
            q.add_argument("--no-answer", action="store_true", help="chỉ in phần truy hồi, không sinh câu trả lời")
        q.add_argument("--mode", choices=["rag", "kg", "both"], default="both")
        q.set_defaults(func=fn)

    sub.add_parser("graph", help="in các bộ ba và xuất storage/kg.html").set_defaults(func=cmd_graph)

    q = sub.add_parser("eval", help="so sánh RAG và KG-RAG trên bộ câu hỏi có đáp án")
    q.add_argument("--cases", default="eval/questions.jsonl")
    q.add_argument("--no-answer", action="store_true", help="chỉ đo recall truy hồi, không sinh câu trả lời")
    q.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(Settings.from_env(), args)


if __name__ == "__main__":
    main()
