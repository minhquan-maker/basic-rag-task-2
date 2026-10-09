import hashlib
import math
import re

import pytest

from ragkg.config import Settings
from ragkg.models import Extraction, Mentions, Triple

P, O, L, A = "Người", "Tổ chức", "Địa điểm", "Giải thưởng"

# "LLM" giả: trả bộ ba cố định theo TIÊU ĐỀ tài liệu, giống một lần trích xuất lý tưởng
SCRIPT = {
    "Alan Turing": [("Alan Turing", P, "sinh_tại", "Luân Đôn", L),
                    ("Giải Turing", A, "đặt_tên_theo", "Alan Turing", P)],
    "ACM": [("Giải Turing", A, "trao_bởi", "ACM", O),
            ("ACM", O, "trụ_sở_tại", "New York", L)],
    "Geoffrey Hinton": [("Geoffrey Hinton", P, "nhận_giải", "Giải Turing", A),
                        ("Hinton", P, "giáo_sư_tại", "Đại học Toronto", O),
                        ("Đại học Toronto", O, "nằm_tại", "Toronto", L)],
    "Yann LeCun": [("Yann LeCun", P, "nhận_giải", "Giải Turing", A)],
}
NAMES_IN_QUESTIONS = ["Hinton", "LeCun", "Turing", "ACM"]


class FakeLLM:
    def __init__(self):
        self.structured_calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return "FAKE"

    def generate_structured(self, prompt, schema, system=None):
        self.structured_calls += 1
        if schema is Mentions:
            q = prompt.split("CÂU HỎI:")[-1]
            return Mentions(entities=[n for n in NAMES_IN_QUESTIONS if n in q])
        title = re.search(r"TIÊU ĐỀ: (.*)", prompt).group(1)
        return Extraction(triples=[Triple(subject=s, subject_type=st, predicate=p, object=o, object_type=ot)
                                   for s, st, p, o, ot in SCRIPT.get(title, [])])


class FakeEmbedder:
    """Băm n-gram ký tự thành vector 64 chiều: văn bản giống nhau thì cosine cao, không cần mạng."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed(self, texts, kind):
        self.embedded.extend(texts)
        out = []
        for t in texts:
            v, s = [0.0] * 64, t.casefold()
            for i in range(len(s) - 2):
                v[int(hashlib.md5(s[i:i + 3].encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_dir=tmp_path / "storage", data_dir=tmp_path / "data",
                    request_interval=0, entity_threshold=0.8)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def emb():
    return FakeEmbedder()
