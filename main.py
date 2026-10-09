import argparse

from src.config import Config
from src.documents import load_documents, make_chunks
from src.kg_builder import KnowledgeGraph


def get_llm(cfg):
    from src.llm import GeminiLLM
    return GeminiLLM(cfg)


def cmd_index(cfg, args):
    from src.kg_rag import KGRAG
    from src.vector_rag import VectorRAG

    docs = load_documents(cfg.data_dir)
    chunks = make_chunks(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"Đọc {len(docs)} tài liệu -> {len(chunks)} đoạn")
    llm = get_llm(cfg)
    if args.only in (None, "rag"):
        print(f"[RAG] đã lưu {VectorRAG(llm, cfg).build_index(chunks)} vector")
    if args.only in (None, "kg"):
        print("[KG] đang trích bộ ba bằng LLM...")
        print(f"[KG] {KGRAG(llm, cfg).build_index(chunks)} -> {cfg.kg_path}")


def show_rag(r):
    print("\n=== RAG thường ===")
    for h in r["contexts"]:
        print(f"  - [{h['id']}] score={h['score']}: {h['text'][:80]}...")
    print("Trả lời:", r["answer"].strip())


def show_kg(r):
    print("\n=== KG-RAG ===")
    print("  Thực thể gốc:", ", ".join(r["entities"]) or "(không có)")
    for h, rel, t in r["triples"]:
        print(f"  - {h} --[{rel}]--> {t}")
    print("Trả lời:", r["answer"].strip())


def ask(cfg, llm, question, mode):
    if mode in ("rag", "both"):
        from src.vector_rag import VectorRAG
        show_rag(VectorRAG(llm, cfg).answer(question))
    if mode in ("kg", "both"):
        from src.kg_rag import KGRAG
        show_kg(KGRAG(llm, cfg).answer(question))


def cmd_ask(cfg, args):
    ask(cfg, get_llm(cfg), args.question, args.mode)


def cmd_chat(cfg, args):
    llm = get_llm(cfg)
    print("Nhập câu hỏi (Enter trống để thoát).")
    while (q := input("\n> ").strip()):
        ask(cfg, llm, q, args.mode)


def cmd_graph(cfg, args):
    from src.visualize import export_html
    if not cfg.kg_path.exists():
        return print("Chưa có đồ thị, chạy `python main.py index` trước.")
    kg = KnowledgeGraph.load(cfg.kg_path)
    print("Đồ thị:", kg.stats())
    for h, t, d in kg.g.edges(data=True):
        print(f"  {kg.name(h)} --[{d['relation']}]--> {kg.name(t)}")
    print("Đã xuất:", export_html(kg, cfg.storage_dir / "kg.html"))


def main():
    parser = argparse.ArgumentParser(description="RAG cơ bản và KG-RAG cơ bản (Gemini)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="xây vector index và/hoặc knowledge graph")
    p.add_argument("--only", choices=["rag", "kg"])
    p.set_defaults(func=cmd_index)

    for name, func, helptext in (("ask", cmd_ask, "hỏi một câu"), ("chat", cmd_chat, "hỏi đáp liên tục")):
        p = sub.add_parser(name, help=helptext)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--mode", choices=["rag", "kg", "both"], default="both")
        p.set_defaults(func=func)

    sub.add_parser("graph", help="in và xuất KG ra HTML").set_defaults(func=cmd_graph)

    args = parser.parse_args()
    args.func(Config(), args)


if __name__ == "__main__":
    main()
