ANSWER_SYSTEM = (
    "Bạn là trợ lý hỏi đáp. Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp; "
    "được phép nối nhiều thông tin trong ngữ cảnh để suy ra đáp án. "
    "Nếu ngữ cảnh không đủ, nói rõ là không tìm thấy, không được bịa. "
    "Trả lời ngắn gọn bằng tiếng Việt."
)

VECTOR_RAG_PROMPT = """NGỮ CẢNH:
{context}

CÂU HỎI: {question}

Trả lời, cuối câu ghi nguồn dạng [tên_file#số_đoạn]."""

KG_RAG_PROMPT = """SỰ KIỆN TỪ ĐỒ THỊ TRI THỨC (thực thể --[quan hệ]--> thực thể):
{triples}
{chunks}
CÂU HỎI: {question}

Suy luận qua chuỗi sự kiện (có thể phải nối nhiều sự kiện) để trả lời.
Cuối câu trả lời nêu ngắn gọn chuỗi sự kiện đã dùng."""

EXTRACT_SYSTEM = "Bạn trích xuất thực thể và quan hệ từ văn bản để xây đồ thị tri thức."

# {{ }} là dấu ngoặc nhọn nguyên văn khi dùng str.format
EXTRACT_PROMPT = """Quy tắc:
- Thực thể: người, tổ chức, địa điểm, giải thưởng, công trình, khái niệm... Dùng tên đầy đủ, nhất quán
  (luôn "Albert Einstein", không viết "ông", "Einstein" lúc này lúc khác).
- Quan hệ: cụm ngắn tiếng Việt, chữ thường, nối bằng gạch dưới (vd: sinh_tại, thuộc_quốc_gia, nhận_giải).
- Chỉ trích thông tin có trong văn bản, không suy diễn.

Trả về JSON:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}

VĂN BẢN:
{text}"""

QUESTION_ENTITY_PROMPT = """Liệt kê các thực thể (tên riêng, địa điểm, tổ chức, giải thưởng, khái niệm) có trong câu hỏi.
Trả về JSON dạng {{"entities": ["...", "..."]}}.

CÂU HỎI: {question}"""
