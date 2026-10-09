import unicodedata

from ragkg.graph import KnowledgeGraph, normalize
from ragkg.resolve import merge_aliases


def test_dedup_triples_and_merge_sources():
    kg = KnowledgeGraph()
    kg.add_triple("Albert Einstein", "sinh tại", "Ulm", "a#0")
    kg.add_triple("albert  einstein", "sinh_tại", "ULM", "b#0")
    assert kg.g.number_of_nodes() == 2 and kg.g.number_of_edges() == 1
    assert list(kg.g.edges(data=True))[0][2]["sources"] == ["a#0", "b#0"]


def test_normalize_is_unicode_safe():
    composed = "Luân Đôn"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed and normalize(composed) == normalize(decomposed)


def test_roundtrip_keeps_aliases(tmp_path):
    kg = KnowledgeGraph()
    kg.add_triple("Hinton", "làm_việc_tại", "Google", "a#0")
    kg.add_triple("Geoffrey Hinton", "sinh_tại", "Luân Đôn", "a#0")
    merge_aliases(kg)
    kg.save(tmp_path / "kg.json")
    kg2 = KnowledgeGraph.load(tmp_path / "kg.json")
    assert kg2.find("Hinton") == kg2.find("geoffrey hinton") is not None


def test_alias_merge_moves_edges():
    kg = KnowledgeGraph()
    kg.add_triple("Geoffrey Hinton", "sinh_tại", "Luân Đôn", "a#0", "Người", "Địa điểm")
    kg.add_triple("Hinton", "làm_việc_tại", "Google", "a#1", "Người", "Tổ chức")
    assert merge_aliases(kg) == [("hinton", "geoffrey hinton")]
    assert kg.g.number_of_nodes() == 3
    assert kg.g.has_edge("geoffrey hinton", "google", key="làm_việc_tại")


def test_alias_merge_skips_ambiguous_and_conflicting_types():
    kg = KnowledgeGraph()
    kg.add_triple("Alan Turing", "nhận_giải", "Giải Turing", "a#0", "Người", "Giải thưởng")
    kg.add_triple("Turing", "x", "Y", "a#0")  # "Turing" ⊂ cả "Alan Turing" lẫn "Giải Turing"
    kg.add_triple("Đại học Toronto", "nằm_tại", "Toronto", "a#0", "Tổ chức", "Địa điểm")
    assert merge_aliases(kg) == []
