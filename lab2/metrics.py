"""Матриця плутанини, вартість помилок, precision/recall.

Усе реалізовано вручну на NumPy (без sklearn.metrics). Правило рішення скрізь
однакове: прогноз "шахрайство" (1), якщо s >= t.
"""

from __future__ import annotations

import math

import numpy as np

COST_FN = 10  # пропущене шахрайство
COST_FP = 1   # хибна тривога


def validate_inputs(y: np.ndarray, s: np.ndarray) -> None:
    """Перевірка форми й значень: мітки лише 0/1, оцінки скінченні."""
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("y та s мають бути одновимірними масивами однакової довжини.")
    if len(y) == 0:
        raise ValueError("Порожня вибірка.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути лише 0 або 1.")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними (без NaN та inf).")


def confusion_counts(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """Повертає (TP, FP, FN, TN) для правила "прогноз = 1, якщо s >= t".

    Порівняння виконується у float64: оцінки float32 точно вкладаються у float64,
    тому поріг, збережений як float64, дає той самий результат, що й під час пошуку.
    t = +inf дає "усі негативні" (жодна скінченна оцінка не >= +inf).
    """
    validate_inputs(y, s)
    pred = s.astype(np.float64) >= float(t)
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    """C = 10*FN + FP. Amount у формулу не входить."""
    return COST_FN * fn + COST_FP * fp


def precision(tp: int, fp: int) -> float:
    """TP/(TP+FP); NaN, якщо знаменник нульовий (немає жодного позитивного прогнозу)."""
    den = tp + fp
    return tp / den if den > 0 else math.nan


def recall(tp: int, fn: int) -> float:
    """TP/(TP+FN); NaN, якщо знаменник нульовий (у вибірці немає позитивних міток)."""
    den = tp + fn
    return tp / den if den > 0 else math.nan
