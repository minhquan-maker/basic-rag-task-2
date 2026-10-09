import hashlib
import math

import chromadb
import pytest

from src.config import Config
from src.documents import chunk_text, load_documents, make_chunks
from src.kg_builder import KnowledgeGraph
from src.kg_rag import KGRAG
from src.vector_rag import VectorRAG

# FakeLLM "trích" các bộ ba cố định khi đoạn văn chứa cụm khóa tương ứng
FACTS = {
    "sinh ngày 14 tháng 3": [("Albert Einstein", "sinh_tại", "Ulm")],
    "thuộc bang": [("Ulm", "thuộc_quốc_gia", "Đức"), ("Ulm", "nằm_bên", "Sông Danube")],
    "sinh năm 1867": [("Marie Curie", "sinh_tại", "Warszawa")],
    "thủ đô": [("Warszawa", "là_thủ_đô_của", "Ba Lan")],
}


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def embed(self, texts, task_type=None):  # băm n-gram ký tự -> vector 64 chiều, chuẩn hóa
        out = []
        for t in texts:
            v, s = [0.0] * 64, t.casefold()
            for i in range(len(s) - 2):
                v[int(hashlib.md5(s[i:i + 3].encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def generate_json(self, prompt, system=None):
        if "CÂU HỎI:" in prompt:
            q = prompt.split("CÂU HỎI:")[-1]
            return {"entities": [n for n in ["Einstein", "Marie Curie"] if n in q]}
        rels = [r for k, ts in FACTS.items() if k in prompt for r in ts]
        return {"entities": [], "relations": [{"head": h, "relation": r, "tail": t} for h, r, t in rels]}

    def generate(self, prompt, system=None, temperature=0.2):
        self.prompts.append(prompt)
        return "FAKE ANSWER"


@pytest.fixture
def cfg(tmp_path):
    return Config(storage_dir=tmp_path, request_delay=0, entity_match_threshold=0.5)


@pytest.fixture
def chunks(cfg):
    return make_chunks(load_documents(cfg.data_dir), cfg.chunk_size, cfg.chunk_overlap)


def test_chunk_size_and_overlap():
    text = " ".join(f"Câu số {i} có nội dung ngắn." for i in range(40))
    parts = chunk_text(text, chunk_size=200, overlap=60)
    assert len(parts) > 1 and all(len(p) <= 200 for p in parts)
    assert parts[0].split(". ")[-1] in parts[1]


def test_graph_dedup_and_roundtrip(tmp_path):
    kg = KnowledgeGraph()
    kg.add_triple("Albert Einstein", "sinh tại", "Ulm", "a#0")
    kg.add_triple("albert  einstein", "sinh_tại", "ulm", "b#0")
    assert kg.g.number_of_nodes() == 2 and kg.g.number_of_edges() == 1
    kg.save(tmp_path / "kg.json")
    edge = list(KnowledgeGraph.load(tmp_path / "kg.json").g.edges(data=True))[0]
    assert edge[2]["sources"] == ["a#0", "b#0"]


def test_vector_rag_retrieve(cfg, chunks):
    rag = VectorRAG(FakeLLM(), cfg, client=chromadb.EphemeralClient())
    rag.build_index(chunks)
    hits = rag.retrieve("Warszawa là thủ đô của nước nào?", k=2)
    assert len(hits) == 2 and hits[0]["id"].startswith("warszawa")


def test_kg_rag_multi_hop(cfg, chunks):
    llm = FakeLLM()
    kgrag = KGRAG(llm, cfg, client=chromadb.EphemeralClient())
    kgrag.build_index(chunks)
    result = kgrag.answer("Einstein sinh ra ở quốc gia nào?")
    assert "Albert Einstein" in result["entities"]  # "Einstein" -> đỉnh qua embedding
    assert ("Albert Einstein", "sinh_tại", "Ulm") in result["triples"]
    assert ("Ulm", "thuộc_quốc_gia", "Đức") in result["triples"]  # đỉnh cách gốc 2 bước
    assert "ĐOẠN VĂN GỐC" in llm.prompts[-1]  # đoạn gốc được đính kèm


def test_kg_rag_without_chunks(cfg, chunks):
    cfg.kg_with_chunks = False
    llm = FakeLLM()
    kgrag = KGRAG(llm, cfg, client=chromadb.EphemeralClient())
    kgrag.build_index(chunks)
    kgrag.answer("Einstein sinh ra ở quốc gia nào?")
    assert "ĐOẠN VĂN GỐC" not in llm.prompts[-1]
