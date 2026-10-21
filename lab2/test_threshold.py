"""Модульні тести пошуку порога та метрик (швидкі, без torch).

    uv run pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import checks
from metrics import confusion_counts, cost, precision, recall
from threshold import candidate_table, select_threshold


def test_three_sets_from_assignment():
    results = checks.run_checks()
    assert all(r["passed"] for r in results.values()), {
        n: r["problems"] for n, r in results.items() if not r["passed"]}


def test_hand_derived_expectations():
    """Ручні очікування (не залежать від жодної реалізації)."""
    def best(name):
        scores, labels = checks.SETS[name]
        return select_threshold(candidate_table(np.array(labels), np.array(scores)))

    assert (best("A")["threshold"], best("A")["C"]) == (0.4, 2)
    assert (best("B")["threshold"], best("B")["C"]) == (math.inf, 0)
    assert (best("C")["threshold"], best("C")["C"]) == (0.8, 10)  # нічия 0.8 vs 0.5 -> більший поріг


@pytest.mark.parametrize("seed", range(300))
def test_random_small_sets_match_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    scores = np.round(rng.random(n), 1)  # багато рівних оцінок
    labels = (rng.random(n) < rng.choice([0.0, 0.1, 0.5, 1.0])).astype(int)  # включно з "усі 0" та "усі 1"
    fast = candidate_table(labels, scores)
    slow = checks.brute_force(scores.tolist(), labels.tolist())
    assert len(fast) == len(slow)
    for i, r in enumerate(slow):
        for col in ("threshold", "TP", "FP", "FN", "TN", "C"):
            assert fast.iloc[i][col] == r[col], (seed, i, col)
    best_fast, best_slow = select_threshold(fast), checks.brute_force_select(slow)
    assert best_fast["threshold"] == best_slow["threshold"] and best_fast["C"] == best_slow["C"]


def test_rule_is_greater_or_equal():
    y = np.array([1, 0])
    s = np.array([0.5, 0.5])
    assert confusion_counts(y, s, 0.5) == (1, 1, 0, 0)  # s == t -> прогноз 1
    assert confusion_counts(y, s, math.inf) == (0, 0, 1, 1)


def test_equal_scores_get_same_prediction():
    y = np.array([1, 0, 0, 1])
    s = np.array([0.7, 0.7, 0.7, 0.1])
    t = candidate_table(y, s)
    assert 0.7 in t["threshold"].tolist()
    r = t[t["threshold"] == 0.7].iloc[0]
    assert (r["TP"], r["FP"]) == (1, 2)  # усі три рівні оцінки одночасно позитивні


def test_float32_scores_roundtrip_exactly():
    """Оцінки зберігаються у float32: поріг у float64 має відтворювати правило точно."""
    rng = np.random.default_rng(0)
    s = rng.random(1000).astype(np.float32)
    s[:50] = np.float32(1.0)  # насичення сигмоїди: реальні тяжкі рівності
    y = (rng.random(1000) < 0.05).astype(int)
    tab = candidate_table(y, s)
    best = select_threshold(tab)
    assert confusion_counts(y, s, float(best["threshold"])) == tuple(int(best[c]) for c in ("TP", "FP", "FN", "TN"))


def test_t_star_depends_only_on_validation():
    rng = np.random.default_rng(1)
    yv, sv = (rng.random(500) < 0.1).astype(int), rng.random(500)
    t1 = select_threshold(candidate_table(yv, sv))["threshold"]
    _ = (rng.random(500) < 0.1).astype(int), rng.random(500)  # "test" не бере участі
    t2 = select_threshold(candidate_table(yv.copy(), sv.copy()))["threshold"]
    assert t1 == t2


def test_precision_recall_undefined_denominators():
    assert math.isnan(precision(0, 0))  # жодного позитивного прогнозу
    assert math.isnan(recall(0, 0))     # у вибірці немає позитивних міток
    assert precision(1, 3) == 0.25 and recall(1, 3) == 0.25


def test_cost_formula():
    assert cost(fn=3, fp=7) == 37


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        candidate_table(np.array([0, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        candidate_table(np.array([0, 1]), np.array([0.1, np.nan]))
