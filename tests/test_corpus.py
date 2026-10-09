from ragkg.corpus import chunk_documents, chunk_text, load_documents


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Câu số {i} có nội dung ngắn." for i in range(40))
    parts = chunk_text(text, size=200, overlap=60)
    assert len(parts) > 1 and all(len(p) <= 200 for p in parts)
    assert parts[0].split(". ")[-1] in parts[1]  # câu cuối đoạn 1 lặp ở đầu đoạn 2


def test_overlong_sentence_is_hard_split():
    parts = chunk_text("a" * 450, size=200, overlap=50)
    assert [len(p) for p in parts] == [200, 200, 50]


def test_title_from_heading_or_filename(tmp_path):
    (tmp_path / "a.md").write_text("# Tiêu đề A\n\nNội dung A.", encoding="utf-8")
    (tmp_path / "ban_b.txt").write_text("Nội dung B.", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("x")
    docs = {d["source"]: d for d in load_documents(tmp_path)}
    assert docs["a.md"]["title"] == "Tiêu đề A" and docs["a.md"]["text"] == "Nội dung A."
    assert docs["ban_b.txt"]["title"] == "ban b" and "skip.pdf" not in docs


def test_chunk_ids_and_embed_text(tmp_path):
    (tmp_path / "x.md").write_text("# X\n\nMột. Hai.", encoding="utf-8")
    [c] = chunk_documents(load_documents(tmp_path), 600, 100)
    assert c.id == "x.md#0" and c.embed_text.startswith("X\n")
