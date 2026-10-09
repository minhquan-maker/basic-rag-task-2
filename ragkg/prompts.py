ANSWER_SYSTEM = (
    "Bạn trả lời câu hỏi chỉ dựa trên NGỮ CẢNH được cung cấp. "
    "Được nối nhiều mẩu thông tin trong ngữ cảnh để suy ra đáp án. "
    "Nếu ngữ cảnh không đủ thì nói rõ là không đủ thông tin, không đoán. "
    "Trả lời ngắn gọn bằng tiếng Việt."
)

VECTOR_PROMPT = """NGỮ CẢNH (mỗi đoạn có mã nguồn trong ngoặc vuông):
{passages}

CÂU HỎI: {question}

Trả lời, cuối câu ghi các mã nguồn đã dùng."""

GRAPH_PROMPT = """SỰ KIỆN TRÍCH TỪ ĐỒ THỊ TRI THỨC (chủ thể --[quan hệ]--> đối tượng):
{facts}

ĐOẠN VĂN GỐC CỦA CÁC SỰ KIỆN:
{passages}

CÂU HỎI: {question}

Nối các sự kiện lại để trả lời; nếu câu hỏi hỏi nhiều đối tượng thì nêu đủ.
Cuối câu ghi chuỗi sự kiện đã dùng."""

EXTRACT_SYSTEM = "Bạn trích bộ ba (chủ thể, quan hệ, đối tượng) từ văn bản để dựng đồ thị tri thức."

EXTRACT_PROMPT = """Quy tắc:
- Chỉ trích điều văn bản nói rõ, không suy diễn, không thêm kiến thức ngoài.
- Dùng tên đầy đủ và nhất quán cho thực thể. Đại từ ("ông", "bà", "nó") phải thay bằng tên thật;
  đoạn văn thuộc tài liệu có tiêu đề bên dưới nên chủ thể ngầm định thường là thực thể trong tiêu đề.
- Quan hệ là cụm ngắn, chữ thường, nối bằng gạch dưới. Ưu tiên bộ từ vựng sau, chỉ tạo quan hệ mới khi không có
  quan hệ nào phù hợp: sinh_tại, mất_tại, sinh_năm, làm_việc_tại, giáo_sư_tại, thành_lập_bởi, thành_lập_năm,
  trụ_sở_tại, thuộc_về, nằm_tại, nằm_bên_sông, thuộc_quốc_gia, nhận_giải, đặt_tên_theo, trao_bởi, phát_triển,
  mua_lại_bởi, đánh_bại, có_quốc_tịch, đồng_nhận_giải_với.
- Mỗi bộ ba gồm hai thực thể; năm, số liệu chỉ làm thực thể khi là đối tượng của quan hệ như sinh_năm, thành_lập_năm.

TIÊU ĐỀ: {title}

VĂN BẢN:
{text}"""

MENTION_PROMPT = """Liệt kê các thực thể (tên riêng, tổ chức, địa điểm, giải thưởng, công trình) xuất hiện trong câu hỏi.
Giữ nguyên cách viết trong câu hỏi.

CÂU HỎI: {question}"""
