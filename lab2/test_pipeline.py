"""Наскрізний тест конвеєра на СИНТЕТИЧНОМУ CSV тієї ж схеми, що й creditcard.csv.

Це лише перевірка коректності коду (поділ, відсутність витоку, відтворюваність,
збереження/завантаження, аналіз). Справжні результати лабораторної - на реальному
датасеті; числа з цього тесту ніде не використовуються.

    uv run pytest -q test_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze
import lab2_starter as L


def make_synthetic_csv(path: Path, n: int = 30_000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    cols = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    y = (rng.random(n) < 0.01).astype(int)
    for k in (1, 3, 5, 7):  # слабкий сигнал у кількох ознаках
        cols[f"V{k}"] = cols[f"V{k}"] + 2.0 * y
    df = pd.DataFrame({"Time": np.arange(n), **cols,
                       "Amount": np.round(rng.lognormal(3.0, 1.2, n), 2), "Class": y})
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab2")
    csv = root / "synthetic.csv"
    make_synthetic_csv(csv)
    tr, va, te, scaler = L.make_splits(csv)
    out = root / "results"
    out.mkdir()
    model = L.train_once(tr)
    L.save_run(out, model, (tr, va, te), scaler, csv)
    return csv, out, (tr, va, te, scaler)


def test_split_is_stratified_disjoint_and_ordered(trained):
    csv, _, (tr, va, te, _) = trained
    df = pd.read_csv(csv)
    ids = np.concatenate([tr.row_ids, va.row_ids, te.row_ids])
    assert np.array_equal(np.sort(ids), np.arange(len(df)))  # покриття, без повторів
    assert len(tr.y) / len(df) == pytest.approx(0.6, abs=1e-3)
    assert len(va.y) / len(df) == pytest.approx(0.2, abs=1e-3)
    pos = [int(s.y.sum()) for s in (tr, va, te)]
    assert abs(pos[1] - pos[2]) <= 1 and abs(pos[0] - 3 * pos[1]) <= 3  # стратифікація
    # сирий Amount збережено окремо й відповідає рядкам CSV
    assert np.array_equal(va.amount, df["Amount"].to_numpy()[va.row_ids])
    assert tr.x.dtype == np.float32


def test_scaler_fit_on_train_only(trained):
    csv, _, (tr, va, te, scaler) = trained
    df = pd.read_csv(csv)
    x = df[L.FEATURES].to_numpy(np.float64)
    assert scaler.n_samples_seen_ == len(tr.y)
    np.testing.assert_allclose(scaler.mean_, x[tr.row_ids].mean(axis=0))
    assert not np.allclose(scaler.mean_, x.mean(axis=0), rtol=0, atol=1e-12)  # не весь набір


def test_saved_predictions_roundtrip(trained):
    _, out, _ = trained
    data = L.load_predictions(out)
    for name in ("validation", "test"):
        s = data[f"{name}_scores"]
        assert s.dtype == np.float32 and ((s >= 0) & (s <= 1)).all()
        assert len(s) == len(data[f"{name}_y"]) == len(data[f"{name}_row_ids"])
    assert "train_scores" not in data


def test_analysis_runs_without_training_and_is_deterministic(trained, tmp_path):
    _, out, _ = trained
    out1, out2 = tmp_path / "a", tmp_path / "b"
    analyze.run(out, out1, readme=None)
    analyze.run(out, out2, readme=None)
    for name in ("threshold.json", "validation_candidates.csv", "results.json",
                 "test_errors_first5.csv", "report.md", "checks.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    for png in ("pr_curve_validation.png", "roc_curve_validation.png"):
        assert (out1 / png).stat().st_size > 5_000

    th = json.loads((out1 / "threshold.json").read_text(encoding="utf-8"))
    data = L.load_predictions(out)
    from threshold import candidate_table, select_threshold
    best = select_threshold(candidate_table(data["validation_y"], data["validation_scores"]))
    assert float(th["t_star"]) == best["threshold"]  # t* визначено лише на validation


def test_test_scores_do_not_influence_threshold(trained, tmp_path):
    _, out, _ = trained
    data = L.load_predictions(out)
    from threshold import candidate_table, select_threshold
    t_ref = select_threshold(candidate_table(data["validation_y"], data["validation_scores"]))["threshold"]
    # будь-яка зміна test-масивів не торкається t*
    rng = np.random.default_rng(0)
    data["test_scores"] = rng.random(len(data["test_scores"])).astype(np.float32)
    t_new = select_threshold(candidate_table(data["validation_y"], data["validation_scores"]))["threshold"]
    assert t_ref == t_new


def test_starter_refuses_nonempty_out(trained, tmp_path):
    csv, out, _ = trained
    import subprocess, sys
    r = subprocess.run([sys.executable, L.__file__, "--data", str(csv), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0
