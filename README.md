# ragkg: Vector RAG và KG-RAG cơ bản

Hai pipeline hỏi đáp trên cùng một bộ tài liệu, để so sánh trực tiếp:

- **Vector RAG**: chia đoạn → embedding → ChromaDB → lấy top-k đoạn gần câu hỏi → LLM trả lời.
- **KG-RAG**: LLM trích bộ ba (chủ thể, quan hệ, đối tượng) từ từng đoạn → đồ thị tri thức (NetworkX) → tìm thực thể trong câu hỏi → lấy đồ thị con quanh các thực thể đó, kèm đoạn văn gốc → LLM trả về câu trả lời.

Không dùng LangChain hay LlamaIndex: mỗi thành phần được viết tay trên thư viện nền để thấy nó làm gì. LLM và embedding hiện dùng Gemini; hai giao diện `LLM` và `Embedder` (`ragkg/protocols.py`) nhỏ nên thay provider chỉ cần viết thêm một lớp.

Khi nào KG-RAG đáng dùng: câu hỏi cần **nối thông tin nằm ở nhiều tài liệu** ("giám đốc của tổ chức tạo ra AlphaGo sinh ở đâu?"). Vector RAG hay lấy được đoạn gần chủ đề nhưng thiếu mắt xích giữa chừng. Với câu hỏi tra cứu trong một đoạn thì vector RAG đủ tốt và rẻ hơn nhiều, vì KG-RAG phải gọi LLM cho mỗi đoạn lúc index.

## Chạy thử

Cần Python ≥ 3.10 và một Gemini API key (https://aistudio.google.com/apikey).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # điền GEMINI_API_KEY

ragkg index                                       # vector index + knowledge graph
ragkg ask "Công ty nào đã mua lại tổ chức tạo ra AlphaGo?"
ragkg ask "..." --mode kg --no-answer             # chỉ xem phần truy hồi, không gọi LLM sinh câu trả lời
ragkg chat
ragkg graph                                       # in bộ ba, xuất storage/kg.html
ragkg eval                                        # so sánh hai pipeline trên eval/questions.jsonl
pytest                                            # 21 test, không cần key
```

`data/` chứa 9 tài liệu ngắn về các nhà nghiên cứu AI (Turing, Hinton, LeCun, Bengio, DeepMind, ...). Muốn dùng tài liệu riêng thì bỏ file `.txt/.md` vào `data/` (dòng đầu `# Tiêu đề` là tiêu đề tài liệu) rồi `ragkg index`.

## Cấu trúc

```
ragkg/
  config.py        Settings, đọc từ .env
  models.py        Chunk, Hit; schema Pydantic cho structured output (Triple, Extraction, Mentions)
  protocols.py     giao diện LLM / Embedder
  providers/gemini.py
  corpus.py        đọc file, chia đoạn theo câu có overlap
  store.py         bọc ChromaDB; sync tăng dần theo hash nội dung
  vector_rag.py    pipeline 1
  graph.py         KnowledgeGraph (MultiDiGraph), chuẩn hóa tên, gộp đỉnh
  resolve.py       gộp tên viết tắt vào tên đầy đủ
  extract.py       trích bộ ba bằng LLM, có cache
  graph_rag.py     pipeline 2: liên kết thực thể, mở rộng đồ thị con, trả lời
  evaluate.py      đo recall nguồn và độ đúng câu trả lời
  viz.py, cli.py
data/  eval/questions.jsonl  tests/
```

## Thư viện

| Thư viện | Dùng để |
|---|---|
| `google-genai` | sinh văn bản, structured output, embedding |
| `pydantic` | schema cho kết quả trích xuất; Gemini ép đầu ra theo schema nên không phải tự parse JSON |
| `chromadb` | vector DB (HNSW, cosine), lưu đĩa |
| `networkx` | cấu trúc đồ thị và duyệt BFS |
| `pyvis` | xem đồ thị trong trình duyệt |
| `python-dotenv`, `pytest` | cấu hình, test |

## Một số quyết định thiết kế

- **Index tăng dần.** Mỗi đoạn lưu hash nội dung (kèm tên model embedding và số chiều); `ragkg index` lần sau chỉ embed đoạn mới hoặc đã sửa và xóa đoạn đã biến mất. Kết quả trích bộ ba được cache theo hash đoạn + model + phiên bản prompt, nên sửa một file chỉ tốn lời gọi LLM cho đoạn đổi. `--fresh` bỏ cache.
- **Tiêu đề đi cùng đoạn.** Tiêu đề tài liệu được ghép vào văn bản khi embedding và đưa vào prompt trích xuất, để đoạn chỉ viết "Ông ..." vẫn gắn đúng chủ thể.
- **Chuẩn hóa tiếng Việt.** Khóa thực thể qua NFC + casefold, vì cùng một chữ có dấu có thể được mã hóa dựng sẵn hoặc dấu rời, và hai dạng đó không bằng nhau khi so chuỗi.
- **Liên kết thực thể 3 tầng** (`GraphRAG.link`): tên/bí danh xuất hiện nguyên văn (ưu tiên cụm dài, nên "Giải Turing" không bị khớp nhầm thành "Alan Turing"); rồi các cụm LLM tách ra khớp chính xác; cuối cùng mới tới embedding với ngưỡng cosine.
- **Gộp tên viết tắt** (`resolve.py`): "Hinton" gộp vào "Geoffrey Hinton" khi là ứng viên duy nhất và loại thực thể không mâu thuẫn. Có chủ đích bỏ qua trường hợp mơ hồ ("Turing" có thể là người hoặc giải thưởng).
- **Mở rộng đồ thị con có kiểm soát.** BFS hai chiều tới `KG_HOPS` bước; đỉnh có hơn `KG_HUB_LIMIT` cạnh không được mở rộng tiếp, để một đỉnh phổ biến không kéo cả đồ thị vào prompt. Cạnh gần gốc và có nhiều bằng chứng được xếp trước khi cắt theo `KG_MAX_FACTS`.
- **Đưa cả đoạn văn gốc** của các bộ ba vào prompt, không chỉ bộ ba, vì bộ ba làm mất sắc thái và chi tiết. Mỗi sự kiện vẫn truy ngược được về đoạn nguồn.

## Đánh giá

`ragkg eval` chạy 10 câu trong `eval/questions.jsonl` (1, 2 và 3 tài liệu cần ghép) qua cả hai pipeline và in theo nhóm:

- **recall nguồn**: tỉ lệ file cần thiết có trong phần truy hồi của pipeline. Không cần LLM chấm.
- **đúng**: câu trả lời có chứa các cụm đáp án (so khớp chuỗi, không phân biệt hoa thường).

Bộ 10 câu này chỉ đủ để thấy xu hướng, không đủ để kết luận thống kê. Chưa có kết quả công bố trong repo vì chưa chạy với Gemini thật khi viết README này; hãy chạy `ragkg eval` và xem `storage/eval.json`.

## Hạn chế

- Chất lượng KG phụ thuộc vào bước trích bộ ba của LLM. Tên quan hệ sinh tự do nên có thể lệch ("sinh_tại" và "sinh_ra_tại"); prompt chỉ gợi ý bộ từ vựng chứ không ép.
- Gộp thực thể là heuristic theo tên: hai thực thể khác nhau, cùng loại, tên lồng nhau (như "Google" và "Google DeepMind") vẫn có thể bị gộp. Tắt bằng `ALIAS_MERGE=false`.
- Mỗi lần index kèm đổi tài liệu, đồ thị được dựng lại từ cache; chưa cập nhật tăng dần từng đỉnh.
- Chunking theo ký tự, không theo token; chưa có reranking, hybrid search (BM25) hay lịch sử hội thoại.
- Lớp gọi Gemini thật (`providers/gemini.py`) chưa có test tự động; test dùng provider giả.

## Hướng phát triển

Lưu đồ thị trong Neo4j và truy vấn bằng Cypher; hybrid search + reranking cho vector RAG; phát hiện cộng đồng và tóm tắt kiểu GraphRAG của Microsoft cho câu hỏi tổng quan; mở rộng bộ đánh giá và dùng LLM làm giám khảo.
