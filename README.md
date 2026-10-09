# RAG cơ bản & KG-RAG cơ bản

Hai chương trình hỏi đáp trên tài liệu `.txt/.md`, cùng dùng Gemini làm LLM + embedding, cài đặt trực tiếp trên các thư viện nền tảng (không dùng LangChain/LlamaIndex) để thấy rõ từng thành phần.

| | RAG thường (vector RAG) | KG-RAG |
|---|---|---|
| Index | chunk → embedding → ChromaDB | chunk → LLM trích bộ ba → Knowledge Graph (NetworkX) |
| Truy hồi | embedding câu hỏi → top-k đoạn gần nhất | tìm thực thể trong câu hỏi → đồ thị con k-hop |
| Mạnh ở | câu hỏi tra cứu một đoạn | câu hỏi cần **nối nhiều tài liệu** (multi-hop) |

## Kiến trúc

```
RAG thường
  Index:  tài liệu ─► chunking ─► embedding ─► ChromaDB (HNSW, cosine)
  Hỏi:    câu hỏi ─► embedding ─► top-k đoạn ─► prompt ─► Gemini ─► trả lời

KG-RAG
  Index:  tài liệu ─► chunking ─► Gemini trích (head, relation, tail) dạng JSON
                    ─► gộp thực thể trùng tên ─► NetworkX MultiDiGraph ─► kg.json
                    └► embedding tên thực thể ─► ChromaDB
  Hỏi:    câu hỏi ─► liên kết thực thể (khớp tên / LLM + embedding)
                  ─► BFS k-hop ─► bộ ba (+ đoạn văn gốc) ─► Gemini ─► trả lời
```

## Cấu trúc

```
main.py               CLI: index / ask / chat / graph
src/config.py         cấu hình (đọc từ .env)
src/llm.py            wrapper Gemini: generate, generate_json, embed (có retry 429/5xx)
src/documents.py      đọc file, chia đoạn theo câu có overlap
src/prompts.py        toàn bộ prompt
src/vector_rag.py     pipeline RAG thường
src/kg_builder.py     KnowledgeGraph (NetworkX) + trích bộ ba bằng LLM
src/kg_rag.py         pipeline KG-RAG: liên kết thực thể, đồ thị con, trả lời
src/visualize.py      xuất đồ thị ra HTML tương tác (pyvis)
data/                 5 tài liệu mẫu (tiếng Việt)
tests/test_offline.py test không cần API key (FakeLLM)
storage/              sinh khi chạy: chroma/, kg.json, chunks.json, kg.html
```

## Thư viện và vai trò

| Thư viện | Vai trò |
|---|---|
| `google-genai` | SDK Gemini: sinh văn bản/JSON và tạo embedding |
| `chromadb` | Vector DB (HNSW, lưu đĩa) cho đoạn văn và tên thực thể |
| `networkx` | Lưu và duyệt Knowledge Graph |
| `pyvis` | Vẽ đồ thị ra HTML |
| `python-dotenv` | Đọc API key/cấu hình từ `.env` |
| `pytest` | Test |

## Cài đặt và chạy

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # điền GEMINI_API_KEY (lấy tại https://aistudio.google.com/apikey)

python main.py index                                   # xây vector index + KG
python main.py ask "Einstein sinh ra ở quốc gia nào?"  # so sánh RAG thường và KG-RAG
python main.py ask "..." --mode kg                     # rag | kg | both
python main.py chat                                    # hỏi đáp liên tục
python main.py graph                                   # in bộ ba, xuất storage/kg.html
```

Muốn dùng tài liệu riêng: bỏ file `.txt/.md` vào `data/` rồi chạy lại `python main.py index`.

Câu hỏi mẫu cần ghép thông tin từ nhiều file:

- Einstein sinh ra ở quốc gia nào? *(einstein → ulm)*
- Thành phố nơi Marie Curie sinh ra nằm bên con sông nào? *(marie_curie → warszawa)*
- Tổ chức nào trao giải Nobel Vật lý mà Einstein nhận năm 1921? *(einstein → nobel)*

## Các tham số chính (`.env`)

| Biến | Ý nghĩa |
|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | độ dài đoạn / phần chồng lấn (ký tự) |
| `TOP_K` | số đoạn lấy về của RAG thường |
| `KG_HOPS` / `KG_MAX_TRIPLES` | độ sâu BFS / số bộ ba tối đa đưa vào prompt |
| `KG_WITH_CHUNKS` | đính kèm đoạn văn gốc của các bộ ba vào prompt (tránh mất chi tiết) |
| `ENTITY_MATCH_THRESHOLD` | ngưỡng cosine để khớp "Einstein" với đỉnh "Albert Einstein" |
| `REQUEST_DELAY` | giây nghỉ giữa các lần gọi LLM lúc xây KG |

## Test

```bash
python -m pytest -q
```

`FakeLLM` thay Gemini nên không cần key. Phủ: chunking, gộp thực thể, lưu/đọc đồ thị, truy hồi vector, truy hồi nhiều bước trên đồ thị, bật/tắt đoạn văn gốc. Phần gọi Gemini thật (`src/llm.py`) chưa được kiểm thử tự động.

## Hạn chế và hướng phát triển

- Xây KG tốn một lần gọi LLM cho mỗi đoạn, nên chậm và tốn quota với kho tài liệu lớn.
- Gộp thực thể chỉ dựa trên tên chuẩn hóa; "Einstein" và "Albert Einstein" có thể thành hai đỉnh nếu LLM viết không nhất quán (prompt đã yêu cầu dùng tên đầy đủ).
- Chất lượng KG phụ thuộc hoàn toàn vào bước trích bộ ba của LLM.
- Mở rộng: Neo4j + Cypher, hybrid search (BM25 + vector) và reranking, phát hiện cộng đồng + tóm tắt kiểu Microsoft GraphRAG, đánh giá tự động (RAGAS).
