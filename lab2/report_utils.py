"""Допоміжне: Markdown-таблиці та вставка згенерованих блоків у README між маркерами."""

from __future__ import annotations

import math
import re
from pathlib import Path


def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["l"] * len(headers)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep[a] for a in align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt_float(x: float, digits: int = 6) -> str:
    """Для відображення (округлення лише тут; у файлах - повна точність)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "не визначено"
    if isinstance(x, float) and math.isinf(x):
        return "+∞" if x > 0 else "−∞"
    return f"{x:.{digits}g}"


def fmt_metric(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "не визначено"
    return f"{x:.4f}"


def inject_block(readme: Path, marker: str, content: str) -> bool:
    """Замінює вміст між <!-- MARKER:START --> та <!-- MARKER:END -->.
    Повертає False, якщо файлу чи маркерів немає."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(<!-- {marker}:START -->)(.*?)(<!-- {marker}:END -->)", flags=re.DOTALL
    )
    if not pattern.search(text):
        return False
    new = pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text)
    readme.write_text(new, encoding="utf-8")
    return True
