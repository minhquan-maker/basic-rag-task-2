from ragkg.evaluate import answer_ok, load_cases, source_recall, summarize


def test_metrics():
    assert source_recall(["a", "b"], {"a"}) == 0.5 and source_recall([], set()) == 1.0
    assert answer_ok("Đó là Luân Đôn.", ["luân đôn"]) and not answer_ok("Paris", ["Luân Đôn"])


def test_shipped_cases_are_valid():
    import pathlib
    cases = load_cases(pathlib.Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl")
    docs = {p.name for p in (pathlib.Path(__file__).resolve().parent.parent / "data").iterdir()}
    assert len(cases) >= 8 and all(set(c.gold_sources) <= docs for c in cases)
    assert {c.hops for c in cases} >= {1, 2, 3}


def test_summarize_groups_by_hops():
    row = lambda h, r, k: {"hops": h, "rag": {"recall": r, "ok": None}, "kg": {"recall": k, "ok": None}}  # noqa: E731
    out = summarize([row(1, 1.0, 1.0), row(2, 0.5, 1.0)])
    assert "1 tài liệu" in out and "2 tài liệu" in out and "tất cả" in out
