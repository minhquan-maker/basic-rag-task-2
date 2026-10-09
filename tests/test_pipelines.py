from ragkg.corpus import chunk_documents
from ragkg.extract import Extractor, build_graph
from ragkg.graph_rag import GraphRAG
from ragkg.models import Mentions
from ragkg.vector_rag import VectorRAG


def write_docs(settings, docs: dict[str, str]):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (settings.data_dir / name).write_text(text, encoding="utf-8")


def chunks_of(settings):
    from ragkg.corpus import load_documents
    return chunk_documents(load_documents(settings.data_dir), settings.chunk_size, settings.chunk_overlap)


DOCS = {
    "turing.md": "# Alan Turing\n\nAlan Turing sinh tại Luân Đôn.",
    "acm.md": "# ACM\n\nACM trao Giải Turing và có trụ sở tại New York.",
    "hinton.md": "# Geoffrey Hinton\n\nHinton là giáo sư tại Đại học Toronto. Ông nhận Giải Turing năm 2018.",
    "lecun.md": "# Yann LeCun\n\nLeCun nhận Giải Turing năm 2018.",
}


def make_graph_rag(settings, llm, emb, docs=DOCS):
    write_docs(settings, docs)
    rag = GraphRAG(llm, emb, settings)
    rag.build(chunks_of(settings))
    return rag


def test_vector_incremental_sync(settings, llm, emb):
    write_docs(settings, DOCS)
    rag = VectorRAG(llm, emb, settings)
    r1 = rag.build(chunks_of(settings))
    assert r1.added == 4 and r1.unchanged == 0
    emb.embedded.clear()
    assert rag.build(chunks_of(settings)).unchanged == 4 and emb.embedded == []  # không embed lại

    (settings.data_dir / "lecun.md").write_text("# Yann LeCun\n\nLeCun sinh năm 1960.", encoding="utf-8")
    (settings.data_dir / "acm.md").unlink()
    r2 = rag.build(chunks_of(settings))
    assert (r2.updated, r2.removed, r2.unchanged) == (1, 1, 2)
    assert len(emb.embedded) == 1


def test_vector_retrieve_ranks_relevant_chunk(settings, llm, emb):
    write_docs(settings, DOCS)
    rag = VectorRAG(llm, emb, settings)
    rag.build(chunks_of(settings))
    top = rag.retrieve("Alan Turing sinh tại Luân Đôn", k=2)
    assert top[0].source == "turing.md" and len(top) == 2


def test_extract_cache_avoids_repeat_llm_calls(settings, llm, emb):
    write_docs(settings, DOCS)
    chunks = chunks_of(settings)
    ex = Extractor(llm, settings.extract_cache_path, "m")
    build_graph(chunks, ex)
    assert ex.calls == 4
    ex2 = Extractor(llm, settings.extract_cache_path, "m")  # đối tượng mới đọc cache từ đĩa
    build_graph(chunks, ex2)
    assert ex2.calls == 0 and ex2.hits == 4
    ex3 = Extractor(llm, settings.extract_cache_path, "m2")  # đổi model -> cache miss
    ex3.extract(chunks[0])
    assert ex3.calls == 1


def test_graph_build_merges_alias_and_persists(settings, llm, emb):
    rag = make_graph_rag(settings, llm, emb)
    assert rag.kg.find("Hinton") == rag.kg.find("Geoffrey Hinton")
    reloaded = GraphRAG(llm, emb, settings)
    assert reloaded.kg.stats() == rag.kg.stats() and reloaded.chunks == rag.chunks


def test_link_prefers_longest_match_and_alias(settings, llm, emb):
    rag = make_graph_rag(settings, llm, emb)
    keys = rag.link("Giải Turing do ai trao?")
    assert [rag.kg.name(k) for k in keys][0] == "Giải Turing"
    assert "alan turing" not in keys  # "Turing" nằm trong "Giải Turing", không được khớp thêm
    assert rag.kg.name(rag.link("Hinton làm gì?")[0]) == "Geoffrey Hinton"  # qua bí danh


def test_embedding_fallback_links_misspelled_entity(settings, llm, emb, monkeypatch):
    rag = make_graph_rag(settings, llm, emb)
    question = "Giáo sư Geoffrey Hintun dạy ở đâu?"  # sai chính tả: không khớp nguyên văn
    assert rag.link(question) == []  # LLM giả không tách được gì
    monkeypatch.setattr(llm, "generate_structured", lambda *a, **k: Mentions(entities=["Geoffrey Hintun"]))
    assert rag.kg.name(rag.link(question)[0]) == "Geoffrey Hinton"  # đỉnh gần nhất theo embedding
    monkeypatch.setattr(llm, "generate_structured", lambda *a, **k: Mentions(entities=["Con mèo"]))
    assert rag.link("Con mèo ở đâu?") == []  # dưới ngưỡng cosine thì không ép khớp


def test_multi_hop_facts_and_source_passages(settings, llm, emb):
    rag = make_graph_rag(settings, llm, emb)
    ans = rag.answer("Giải Turing do tổ chức nào trao và trụ sở ở đâu?")
    facts = {str(f) for f in ans.context.facts}
    assert "Giải Turing --[trao_bởi]--> ACM" in facts
    assert "ACM --[trụ_sở_tại]--> New York" in facts  # hop thứ hai
    assert "acm.md" in ans.sources
    assert "[acm.md#0]" in llm.prompts[-1]  # đoạn văn gốc được đưa vào prompt


def test_hops_limit_and_hub_limit(settings, llm, emb):
    rag = make_graph_rag(settings, llm, emb)
    seeds = rag.link("Hinton")
    one_hop = {str(f) for f in GraphRAG.expand(_with(rag, kg_hops=1), seeds)}
    assert "Giải Turing --[trao_bởi]--> ACM" not in one_hop  # cách Hinton 2 bước
    assert "Geoffrey Hinton --[nhận_giải]--> Giải Turing" in one_hop
    two_hop = {str(f) for f in GraphRAG.expand(_with(rag, kg_hops=2), seeds)}
    assert "Giải Turing --[trao_bởi]--> ACM" in two_hop
    # Giải Turing có 4 cạnh; hub_limit=1 chặn việc đi tiếp qua nó
    hubbed = {str(f) for f in GraphRAG.expand(_with(rag, kg_hops=3, kg_hub_limit=1), seeds)}
    assert "Giải Turing --[trao_bởi]--> ACM" not in hubbed


def test_unknown_question_reports_nothing_found(settings, llm, emb):
    rag = make_graph_rag(settings, llm, emb)
    ans = rag.answer("Con mèo của tôi tên gì?")
    assert not ans.context.facts and "Không tìm thấy" in ans.text and llm.prompts == []


def _with(rag, **kw):
    from dataclasses import replace
    rag.s = replace(rag.s, **kw)
    return rag
