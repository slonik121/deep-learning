"""Пошук порога за мінімумом C(t) = 10*FN(t) + FP(t): сортування + кумулятивні суми.

Складність O(n log n): одне сортування, далі лише cumsum та векторні операції.
Жодного проходу по вибірці для кожного порога.

Ідея. Нехай оцінки відсортовано за спаданням. Прогноз "1" при порозі t = s_(k)
отримують усі об'єкти з s >= s_(k), тобто перші k' позицій, де k' - остання
позиція групи рівних оцінок. Тому
    TP(t) = cumsum(y)[k'],   FP(t) = cumsum(1 - y)[k'],
    FN(t) = P - TP(t),       TN(t) = N - FP(t).
Для t = +inf: TP = FP = 0, FN = P, TN = N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import COST_FN, COST_FP, validate_inputs

COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Повна таблиця кандидатів: +inf та всі унікальні оцінки, за спаданням порога.

    Рівні оцінки утворюють один кандидат: беремо кумулятивні суми в ОСТАННІЙ
    позиції кожної групи рівних значень, тож рівні оцінки завжди отримують
    однаковий прогноз.
    """
    validate_inputs(y, s)
    s64 = s.astype(np.float64)  # float32 -> float64 без втрат
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    order = np.argsort(-s64, kind="stable")  # за спаданням оцінки
    s_sorted = s64[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    # остання позиція кожної групи рівних оцінок
    is_group_end = np.empty(len(s_sorted), dtype=bool)
    is_group_end[:-1] = s_sorted[1:] != s_sorted[:-1]
    is_group_end[-1] = True
    idx = np.flatnonzero(is_group_end)

    thresholds = np.concatenate(([np.inf], s_sorted[idx]))
    tp = np.concatenate(([0], tp_cum[idx]))
    fp = np.concatenate(([0], fp_cum[idx]))
    fn = n_pos - tp
    tn = n_neg - fp
    c = COST_FN * fn + COST_FP * fp
    return pd.DataFrame(
        {"threshold": thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c},
        columns=COLUMNS,
    )


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок таблиці з мінімальною C; за рівної вартості - найбільший поріг.

    +inf теж кандидат, тож якщо "усі негативні" мають мінімальну вартість
    (наприклад, коли позитивів немає), t* = +inf.
    """
    best_cost = table["C"].min()
    t_star = table.loc[table["C"] == best_cost, "threshold"].max()
    row = table.loc[table["threshold"] == t_star]
    assert len(row) == 1, "поріг має бути унікальним у таблиці кандидатів"
    return row.iloc[0]


def row_at(table: pd.DataFrame, t: float) -> pd.Series:
    """Рядок кандидата, що відповідає порогу t (будь-який t: беремо найменший
    кандидат >= t, бо правило s >= t для такого кандидата дає той самий прогноз)."""
    cands = table.loc[table["threshold"] >= t, "threshold"]
    return table.loc[table["threshold"] == cands.min()].iloc[0]
