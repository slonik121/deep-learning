"""Пункт 3: незалежна перевірка пошуку порога на малих наборах A, B, C.

Еталон `brute_force` навмисно НЕ використовує ні threshold.py, ні metrics.py,
ні NumPy-сортування: чисті цикли Python, кандидати з set(), правило s >= t
застосовується до кожного кандидата й кожного об'єкта окремо. Повільний (O(n^2)),
тому дозволений лише для цих малих наборів.

Крім порівняння "швидкий vs повільний" перевіряються ВРУЧНУ виведені очікування
(`EXPECTED`): вони не залежать від жодної з двох реалізацій.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from threshold import candidate_table, select_threshold

INF = math.inf

# Набори з умови завдання. C: 0,8; одинадцять разів 0,5; 0,2.
SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}

# Виведено вручну (див. README, розділ "Перевірка"): (t*, C(t*), кількість кандидатів,
# вартості по кандидатах за спаданням порога).
EXPECTED: dict[str, dict] = {
    "A": {"t_star": 0.4, "C_min": 2, "n_candidates": 6,
          "costs": [20, 21, 12, 2, 3, 4]},
    "B": {"t_star": INF, "C_min": 0, "n_candidates": 4,
          "costs": [0, 1, 2, 3]},
    "C": {"t_star": 0.8, "C_min": 10, "n_candidates": 4,
          "costs": [20, 10, 10, 11]},
}


def brute_force(scores: list[float], labels: list[int]) -> list[dict]:
    """Прямий перебір: для кожного кандидата порахувати матрицю плутанини
    безпосереднім застосуванням s_i >= t до кожного об'єкта."""
    candidates = sorted(set(scores) | {INF}, reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s_i, y_i in zip(scores, labels):
            predicted_fraud = s_i >= t
            if predicted_fraud and y_i == 1:
                tp += 1
            elif predicted_fraud and y_i == 0:
                fp += 1
            elif (not predicted_fraud) and y_i == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "C": 10 * fn + fp})
    return rows


def brute_force_select(rows: list[dict]) -> dict:
    """Мінімум C; за рівності - найбільший поріг (інша реалізація, ніж у threshold.py)."""
    best = None
    for r in rows:
        if best is None or r["C"] < best["C"] or (r["C"] == best["C"] and r["threshold"] > best["threshold"]):
            best = r
    return best


def compare_one(name: str) -> dict:
    scores, labels = SETS[name]
    slow_rows = brute_force(scores, labels)
    slow_best = brute_force_select(slow_rows)

    fast = candidate_table(np.array(labels), np.array(scores))
    fast_best = select_threshold(fast)

    problems: list[str] = []
    # 1) кількість і порядок кандидатів, усі лічильники та вартості - точний збіг
    if len(fast) != len(slow_rows):
        problems.append(f"кількість кандидатів: швидкий {len(fast)} vs повільний {len(slow_rows)}")
    else:
        for i, r in enumerate(slow_rows):
            f = fast.iloc[i]
            for col in ("threshold", "TP", "FP", "FN", "TN", "C"):
                if f[col] != r[col]:
                    problems.append(f"рядок {i}, {col}: швидкий {f[col]} vs повільний {r[col]}")
    # 2) обраний поріг і правило нічиєї
    if fast_best["threshold"] != slow_best["threshold"] or fast_best["C"] != slow_best["C"]:
        problems.append(
            f"t*: швидкий {fast_best['threshold']} (C={fast_best['C']}) vs "
            f"повільний {slow_best['threshold']} (C={slow_best['C']})"
        )
    # 3) ручні очікування
    exp = EXPECTED[name]
    if fast_best["threshold"] != exp["t_star"] or int(fast_best["C"]) != exp["C_min"]:
        problems.append(f"t*/C не збігаються з ручним очікуванням {exp['t_star']}/{exp['C_min']}")
    if len(fast) != exp["n_candidates"] or fast["C"].tolist() != exp["costs"]:
        problems.append(f"вартості кандидатів {fast['C'].tolist()} ≠ очікувані {exp['costs']}")

    return {
        "name": name,
        "passed": not problems,
        "problems": problems,
        "n_objects": len(scores),
        "n_candidates": len(slow_rows),
        "t_star": slow_best["threshold"],
        "C_min": slow_best["C"],
        "table": pd.DataFrame(slow_rows),
        "tie_candidates": [r["threshold"] for r in slow_rows if r["C"] == slow_best["C"]],
    }


def run_checks() -> dict[str, dict]:
    return {name: compare_one(name) for name in SETS}


def fmt_t(t: float) -> str:
    return "+∞" if t == INF else f"{t:g}"


def checks_markdown(results: dict[str, dict]) -> str:
    """Таблиці перевірки у Markdown (для README та results/checks.md)."""
    lines = []
    for name, r in results.items():
        scores, labels = SETS[name]
        lines.append(f"**Набір {name}** ({r['n_objects']} об'єктів, "
                     f"{sum(labels)} позитивних), {'✅ збіг' if r['passed'] else '❌ РОЗБІЖНІСТЬ'}\n")
        lines.append("| t | TP | FP | FN | TN | C |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in r["table"].iterrows():
            mark = " ← t*" if row["threshold"] == r["t_star"] else ""
            lines.append(f"| {fmt_t(row['threshold'])}{mark} | {int(row['TP'])} | {int(row['FP'])} | "
                         f"{int(row['FN'])} | {int(row['TN'])} | {int(row['C'])} |")
        ties = ", ".join(fmt_t(t) for t in r["tie_candidates"])
        lines.append(f"\nМінімум C = {r['C_min']} на порогах: {ties}; обрано t* = **{fmt_t(r['t_star'])}**.\n")
    return "\n".join(lines)


def save_checks(results: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {
            "passed": r["passed"],
            "problems": r["problems"],
            "t_star": "+inf" if r["t_star"] == INF else r["t_star"],
            "C_min": r["C_min"],
            "tie_candidates": ["+inf" if t == INF else t for t in r["tie_candidates"]],
            "table": [
                {**{k: ("+inf" if v == INF else v) for k, v in row.items()}}
                for row in r["table"].to_dict(orient="records")
            ],
        }
        for name, r in results.items()
    }
    (out_dir / "checks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8"
    )
    (out_dir / "checks.md").write_text(checks_markdown(results), encoding="utf-8")


def main() -> None:
    results = run_checks()
    print(checks_markdown(results))
    failed = [n for n, r in results.items() if not r["passed"]]
    if failed:
        for n in failed:
            print(f"НАБІР {n}:", *results[n]["problems"], sep="\n  - ")
        raise SystemExit(1)
    print("Усі три перевірки пройдено.")


if __name__ == "__main__":
    main()
