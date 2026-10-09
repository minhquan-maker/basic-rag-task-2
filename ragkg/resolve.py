import re

from .graph import KnowledgeGraph


def _tokens(key: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", key))


def _compatible(a: str, b: str) -> bool:
    return not a or not b or a == b


def merge_aliases(kg: KnowledgeGraph) -> list[tuple[str, str]]:
    """Gộp tên viết tắt vào tên đầy đủ ("Hinton" -> "Geoffrey Hinton").

    Đỉnh A được gộp vào đỉnh B khi các từ của A là tập con thực sự của các từ của B,
    loại thực thể không mâu thuẫn, và B là ứng viên DUY NHẤT. Nếu có nhiều ứng viên
    ("Turing" có thể là "Alan Turing" hoặc "Giải Turing") thì bỏ qua thay vì đoán.
    Đây là heuristic: hai thực thể thật sự khác nhau nhưng tên lồng nhau và cùng loại
    (vd "Google" và "Google DeepMind") vẫn có thể bị gộp; tắt bằng ALIAS_MERGE=false.
    """
    nodes = {k: (_tokens(k), d["type"]) for k, d in kg.g.nodes(data=True)}
    plan: dict[str, str] = {}
    for a, (ta, type_a) in nodes.items():
        cands = [b for b, (tb, type_b) in nodes.items()
                 if b != a and ta < tb and _compatible(type_a, type_b)]
        if len(cands) == 1:
            plan[a] = cands[0]
    merged = []
    for a, b in plan.items():
        while b in plan:  # B cũng bị gộp vào C thì đi theo chuỗi
            b = plan[b]
        if a in kg.g and b in kg.g and a != b:
            kg.merge(a, b)
            merged.append((a, b))
    return merged
