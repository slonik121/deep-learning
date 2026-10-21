"""Пункти 2–4: аналіз порогів зі ЗБЕРЕЖЕНИХ прогнозів (без навчання й без CSV).

    uv run python analyze.py --saved results

Порядок роботи гарантує відсутність витоку з test:
  1) спершу виконуються три перевірки пошуку порога (checks.py); при невдачі - вихід;
  2) на validation будуються криві, таблиця кандидатів і обирається t*;
     t* записується у threshold.json ДО того, як хоч один масив test_* буде прочитано
     для підрахунків;
  3) test оцінюється з порогом, ПРОЧИТАНИМ назад із threshold.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import checks  # noqa: E402
from lab2_starter import load_predictions  # noqa: E402
from metrics import COST_FN, COST_FP, confusion_counts, cost, precision, recall  # noqa: E402
from report_utils import fmt_float, fmt_metric, inject_block, md_table  # noqa: E402
from threshold import candidate_table, row_at, select_threshold  # noqa: E402

HERE = Path(__file__).resolve().parent
STANDARD_T = 0.5
ELKAN_T = COST_FP / (COST_FP + COST_FN)  # 1/11: теоретичний поріг для відкаліброваних ймовірностей
EXPECTED_TOTAL, EXPECTED_FRAUD = 284_807, 492  # з умови завдання

# Палітра: одна нейтрально-синя крива, дві контрастні робочі точки (колір + форма маркера).
C_CURVE, C_STD, C_STAR, C_GRID = "#3b6fb6", "#d9822b", "#1b1b1b", "#d9d9d9"


# --------------------------------------------------------------------------- helpers
def _row_dict(t: float, tp: int, fp: int, fn: int, tn: int) -> dict:
    return {"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def _t_to_json(t: float) -> str:
    return repr(float(t))  # 'inf' або повна точність; float() відновлює точно


def _t_from_json(s: str) -> float:
    return float(s)


def split_summary_table(data: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "test"):
        y = data[f"{name}_y"]
        pos = int(np.count_nonzero(y == 1))
        rows.append({"split": name, "n": len(y), "positives": pos,
                     "positive_fraction": pos / len(y)})
    return pd.DataFrame(rows)


def integrity_notes(data: dict[str, np.ndarray], saved_dir: Path) -> list[str]:
    """Перевірки поділу: непересічність, повне покриття, відповідність умові, scaler лише на train."""
    notes = []
    ids = [data[f"{n}_row_ids"] for n in ("train", "validation", "test")]
    all_ids = np.concatenate(ids)
    total = len(all_ids)
    disjoint = len(np.unique(all_ids)) == total
    covers = disjoint and np.array_equal(np.sort(all_ids), np.arange(total))
    notes.append(f"Індекси train/validation/test не перетинаються й покривають усі {total} рядків CSV: "
                 f"{'так' if covers else 'НІ'}.")
    n_pos = sum(int(np.count_nonzero(data[f'{n}_y'] == 1)) for n in ("train", "validation", "test"))
    ok = total == EXPECTED_TOTAL and n_pos == EXPECTED_FRAUD
    notes.append(f"Усього {total} транзакцій, шахрайських {n_pos} "
                 f"(очікувано за умовою: {EXPECTED_TOTAL} та {EXPECTED_FRAUD}): {'збігається' if ok else 'НЕ збігається'}.")
    prep = saved_dir / "preprocessing.npz"
    if prep.exists():
        with np.load(prep, allow_pickle=False) as p:
            seen = int(p["n_samples_seen"])
        n_train = len(data["train_y"])
        notes.append(f"StandardScaler бачив {seen} об'єктів; розмір train = {n_train}: "
                     f"{'навчений лише на train' if seen == n_train else 'УВАГА: розміри не збігаються'}.")
    return notes


# --------------------------------------------------------------------------- plots
def plot_pr(y: np.ndarray, s: np.ndarray, rows: dict[str, dict], path: Path, ap: float) -> None:
    prec, rec, _ = precision_recall_curve(y, s)
    base = y.mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.step(rec, prec, where="post", color=C_CURVE, lw=2, label=f"PR-крива (AP = {ap:.4f})")
    ax.axhline(base, color="#888888", ls="--", lw=1, label=f"частка шахрайств = {base:.4f}")
    for key, marker, color, label in (("std", "o", C_STD, f"t = {STANDARD_T}"),
                                      ("star", "D", C_STAR, "t*")):
        r = rows[key]
        p, rc = precision(r["TP"], r["FP"]), recall(r["TP"], r["FN"])
        if not (math.isnan(p) or math.isnan(rc)):
            ax.plot(rc, p, marker, color=color, ms=9, mec="white", mew=1.5, ls="none",
                    label=f"{label}: precision={p:.3f}, recall={rc:.3f}")
        else:  # немає жодного позитивного прогнозу: precision не визначена, точки на кривій немає
            ax.plot([], [], marker, color=color, ms=9, mec="white", mew=1.5, ls="none",
                    label=f"{label}: precision не визначена (TP+FP=0)")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)",
           xlim=(0, 1.01), ylim=(0, 1.02),
           title="Крива точність–повнота, validation")
    ax.grid(color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y: np.ndarray, s: np.ndarray, rows: dict[str, dict], path: Path, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y, s)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.plot(fpr, tpr, color=C_CURVE, lw=2, label=f"ROC-крива (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1, label="випадковий класифікатор")
    for key, marker, color, label in (("std", "o", C_STD, f"t = {STANDARD_T}"),
                                      ("star", "D", C_STAR, "t*")):
        r = rows[key]
        ax.plot(r["FP"] / n_neg, r["TP"] / n_pos, marker, color=color, ms=9, mec="white",
                mew=1.5, ls="none",
                label=f"{label}: FPR={r['FP'] / n_neg:.4f}, TPR={r['TP'] / n_pos:.3f}")
    ax.set(xlabel="Частка хибних тривог (FPR)", ylabel="Повнота (TPR)",
           xlim=(-0.005, 1.0), ylim=(0, 1.02), title="ROC-крива, validation")
    ax.grid(color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- main analysis
def run(saved_dir: Path, out_dir: Path, readme: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) Три перевірки. До test переходимо лише після їх проходження.
    check_results = checks.run_checks()
    checks.save_checks(check_results, out_dir)
    failed = [n for n, r in check_results.items() if not r["passed"]]
    if failed:
        raise SystemExit(f"Перевірки пошуку порога не пройдено для наборів {failed}; "
                         f"test не оцінюється. Деталі: {out_dir / 'checks.json'}")
    print("Перевірки A/B/C пройдено.")

    data = load_predictions(saved_dir)
    summary = split_summary_table(data)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    notes = integrity_notes(data, saved_dir)

    # (2) Validation: криві, таблиця кандидатів, t*.
    y_val, s_val = data["validation_y"], data["validation_scores"]
    table = candidate_table(y_val, s_val)
    table.to_csv(out_dir / "validation_candidates.csv", index=False)  # повна точність float64
    best = select_threshold(table)
    t_star = float(best["threshold"])

    p_val, n_val = int(y_val.sum()), int(len(y_val) - y_val.sum())
    tp, fp, fn, tn = confusion_counts(y_val, s_val, STANDARD_T)
    row_std = _row_dict(STANDARD_T, tp, fp, fn, tn)
    # узгодженість: незалежний підрахунок для 0.5 має збігатися з відповідним кандидатом
    cand_std = row_at(table, STANDARD_T)
    assert (int(cand_std["TP"]), int(cand_std["FP"]), int(cand_std["FN"]), int(cand_std["TN"])) == (tp, fp, fn, tn)
    row_star = _row_dict(t_star, *(int(best[c]) for c in ("TP", "FP", "FN", "TN")))
    # узгодженість t*: пряме застосування правила s >= t* до validation
    assert confusion_counts(y_val, s_val, t_star) == (row_star["TP"], row_star["FP"], row_star["FN"], row_star["TN"])
    row_inf = _row_dict(math.inf, 0, 0, p_val, n_val)
    assert confusion_counts(y_val, s_val, math.inf) == (0, 0, p_val, n_val)
    rows = {"std": row_std, "star": row_star, "inf": row_inf}

    ties = table.loc[table["C"] == best["C"], "threshold"].tolist()
    auc = float(roc_auc_score(y_val, s_val))
    ap = float(average_precision_score(y_val, s_val))
    plot_pr(y_val, s_val, rows, out_dir / "pr_curve_validation.png", ap)
    plot_roc(y_val, s_val, rows, out_dir / "roc_curve_validation.png", auc)

    elkan_row = _row_dict(ELKAN_T, *confusion_counts(y_val, s_val, ELKAN_T))

    # Фіксація порога ДО звернення до test.
    (out_dir / "threshold.json").write_text(json.dumps({
        "t_star": _t_to_json(t_star),
        "t_star_float64_hex": t_star.hex() if math.isfinite(t_star) else "inf",
        "rule": "predict fraud (1) iff s >= t",
        "cost": "C = 10*FN + FP",
        "selected_on": "validation",
        "tie_break": "largest threshold among minima of C",
        "validation": {k: (int(v) if k != "threshold" else _t_to_json(v)) for k, v in row_star.items()},
        "n_candidates": int(len(table)),
        "n_thresholds_with_min_cost": len(ties),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # (3) Test: поріг читаємо з файлу.
    t_fixed = _t_from_json(json.loads((out_dir / "threshold.json").read_text(encoding="utf-8"))["t_star"])
    assert t_fixed == t_star
    y_te, s_te = data["test_y"], data["test_scores"]
    amount_te, ids_te = data["test_amount"], data["test_row_ids"]
    te = {}
    for key, t in (("star", t_fixed), ("std", STANDARD_T), ("inf", math.inf)):
        te[key] = _row_dict(t, *confusion_counts(y_te, s_te, t))
    prec_te = precision(te["star"]["TP"], te["star"]["FP"])
    rec_te = recall(te["star"]["TP"], te["star"]["FN"])

    s64 = s_te.astype(np.float64)
    pred = s64 >= t_fixed
    err = pd.DataFrame({"csv_row": ids_te, "label": y_te, "score": s64, "amount": amount_te})
    err["distance_to_threshold"] = err["score"] - t_fixed
    fp_all = err[pred & (y_te == 0)].assign(error="FP").sort_values("csv_row")
    fn_all = err[(~pred) & (y_te == 1)].assign(error="FN").sort_values("csv_row")
    cols = ["csv_row", "error", "label", "score", "amount", "distance_to_threshold"]
    first = pd.concat([fp_all.head(5), fn_all.head(5)])[cols]
    first.to_csv(out_dir / "test_errors_first5.csv", index=False)
    pd.concat([fp_all, fn_all])[cols].to_csv(out_dir / "test_errors_all.csv", index=False)

    result = {
        "validation": {k: {kk: (_t_to_json(vv) if kk == "threshold" else int(vv)) for kk, vv in v.items()}
                       for k, v in rows.items()},
        "validation_roc_auc": auc, "validation_average_precision": ap,
        "test": {k: {kk: (_t_to_json(vv) if kk == "threshold" else int(vv)) for kk, vv in v.items()}
                 for k, v in te.items()},
        "test_precision_at_t_star": None if math.isnan(prec_te) else prec_te,
        "test_recall_at_t_star": None if math.isnan(rec_te) else rec_te,
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # (4) Звіт.
    report = build_report(summary, notes, rows, ties, len(table), auc, ap, elkan_row, te,
                          prec_te, rec_te, first, fp_all, fn_all, err, t_fixed, p_val)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    if readme is not None:
        ok1 = inject_block(readme, "RESULTS", report)
        ok2 = inject_block(readme, "CHECKS", checks.checks_markdown(check_results))
        print(f"README оновлено: RESULTS={'так' if ok1 else 'ні'}, CHECKS={'так' if ok2 else 'ні'}")
    print(report)


# --------------------------------------------------------------------------- report text
def _conf_rows(r: dict) -> str:
    return md_table(
        ["", "Прогноз 1 (s ≥ t)", "Прогноз 0 (s < t)"],
        [["**Факт 1** (шахрайство)", f"TP = {r['TP']}", f"FN = {r['FN']}"],
         ["**Факт 0** (легітимна)", f"FP = {r['FP']}", f"TN = {r['TN']}"]],
        ["l", "r", "r"])


def _stats(df: pd.DataFrame, col: str) -> str:
    if df.empty:
        return "—"
    return f"мін {df[col].min():.2f} / медіана {df[col].median():.2f} / макс {df[col].max():.2f}"


def build_report(summary, notes, rows, ties, n_cand, auc, ap, elkan, te, prec_te, rec_te,
                 first, fp_all, fn_all, err, t_star, p_val) -> str:
    L: list[str] = []
    a = L.append

    a("#### 1. Поділ даних\n")
    a(md_table(["Вибірка", "Об'єктів", "Шахрайських", "Частка шахрайств"],
               [[r.split, r.n, r.positives, f"{100 * r.positive_fraction:.4f} %"]
                for r in summary.itertuples()], ["l", "r", "r", "r"]))
    a("")
    for n in notes:
        a(f"- {n}")
    a("")

    a("#### 2. Validation: криві, ROC-AUC, поріг\n")
    a(f"ROC-AUC на validation = **{auc:.4f}**; середня точність (AP, площа під PR-кривою) = {ap:.4f}; "
      f"частка шахрайств у validation = {p_val / (p_val + rows['inf']['TN']):.4f} "
      f"(це рівень PR-кривої для випадкового класифікатора).\n")
    a("![PR-крива](results/analysis/pr_curve_validation.png)\n")
    a("![ROC-крива](results/analysis/roc_curve_validation.png)\n")
    a(f"Кандидатів порога: {n_cand} (унікальні оцінки validation та +∞). "
      f"Повна таблиця: `results/analysis/validation_candidates.csv`.\n")

    def trow(label, r, tname):
        return [label, tname, r["TP"], r["FP"], r["FN"], r["TN"], r["C"]]
    a(md_table(["Правило на validation", "Поріг", "TP", "FP", "FN", "TN", "C"],
               [trow("Стандартний поріг", rows["std"], "0.5"),
                trow("Мінімум вартості", rows["star"], f"t* = {fmt_float(rows['star']['threshold'], 8)}"),
                trow("Усі негативні", rows["inf"], "+∞")],
               ["l", "r", "r", "r", "r", "r", "r"]))
    a("")
    if len(ties) > 1:
        a(f"Мінімум C = {rows['star']['C']} досягається на {len(ties)} порогах; за правилом нічиєї обрано найбільший "
          f"(t* = {fmt_float(t_star, 8)}).\n")
    else:
        a(f"Мінімум C = {rows['star']['C']} досягається на єдиному порозі t*.\n")
    a(f"Довідково: теоретичний поріг Елкана $p^* = c_{{FP}}/(c_{{FP}}+c_{{FN}}) = 1/11 \\approx {ELKAN_T:.4f}$ "
      f"(коректний лише для відкаліброваних ймовірностей) дає на validation C = {elkan['C']} "
      f"(FP = {elkan['FP']}, FN = {elkan['FN']}). Він **не** використовувався для вибору t*.\n")

    a("#### 3. Test (поріг t* зафіксовано на validation)\n")
    a(f"t* = {fmt_float(t_star, 8)} (повна точність - у `results/analysis/threshold.json`).\n")
    a(_conf_rows(te["star"]))
    a("")
    a(f"- C(test) = **{te['star']['C']}** (= 10·{te['star']['FN']} + {te['star']['FP']})")
    a(f"- precision = TP/(TP+FP) = {te['star']['TP']}/{te['star']['TP'] + te['star']['FP']} = {fmt_metric(prec_te)}"
      + (" (знаменник нульовий: жодної позитивної передбаченої транзакції)" if math.isnan(prec_te) else ""))
    a(f"- recall = TP/(TP+FN) = {te['star']['TP']}/{te['star']['TP'] + te['star']['FN']} = {fmt_metric(rec_te)}"
      + (" (знаменник нульовий: у test немає шахрайських транзакцій)" if math.isnan(rec_te) else ""))
    a("")
    a("Довідково, ті самі правила на test (поріг **не** змінювався після перегляду цих чисел):\n")
    a(md_table(["Правило", "TP", "FP", "FN", "TN", "C"],
               [["t = 0.5", *(te["std"][k] for k in ("TP", "FP", "FN", "TN", "C"))],
                ["t*", *(te["star"][k] for k in ("TP", "FP", "FN", "TN", "C"))],
                ["+∞ (усі негативні)", *(te["inf"][k] for k in ("TP", "FP", "FN", "TN", "C"))]],
               ["l", "r", "r", "r", "r", "r"]))
    a("")

    a("#### 4. Перші п'ять FP та FN на test (за номером рядка CSV)\n")
    if first.empty:
        a("Помилок немає.\n")
    else:
        a(md_table(["Рядок CSV (від 0, без заголовка)", "Тип помилки", "Мітка", "Оцінка s", "Amount", "s − t*"],
                   [[int(r.csv_row), r.error, int(r.label), f"{r.score:.6g}", f"{r.amount:.2f}",
                     f"{r.distance_to_threshold:+.3g}"] for r in first.itertuples()],
                   ["r", "c", "r", "r", "r", "r"]))
        a("")
    fp5, fn5 = first[first.error == "FP"], first[first.error == "FN"]
    a(f"Усього на test: FP = {len(fp_all)}, FN = {len(fn_all)}. Amount (мін / медіана / макс): "
      f"перші FP - {_stats(fp5, 'amount')}; перші FN - {_stats(fn5, 'amount')}; "
      f"усі FP - {_stats(fp_all, 'amount')}; усі FN - {_stats(fn_all, 'amount')}; "
      f"уся test-вибірка - медіана {np.median(err['amount']):.2f}. "
      f"Відстань |s − t*| (медіана): перші FP - {fp5['distance_to_threshold'].abs().median():.3g}, "
      f"перші FN - {fn5['distance_to_threshold'].abs().median():.3g}.\n"
      if len(first) else "")

    a("#### 5. Що змінив вибір порога (числа)\n")
    v_std, v_star = rows["std"], rows["star"]
    t_std, t_star_ = te["std"], te["star"]

    def delta(new, old):
        d = new - old
        return f"{old} → {new} ({d:+d})"
    word_v = ("зменшив" if v_star["C"] < v_std["C"] else "не змінив" if v_star["C"] == v_std["C"] else "збільшив")
    word_t = ("зменшив" if t_star_["C"] < t_std["C"] else "не змінив" if t_star_["C"] == t_std["C"] else "збільшив")
    a(f"- Модель і її оцінки не змінювались; змінилось лише правило рішення: 0.5 → t*.")
    a(f"- **Validation:** вибір t* {word_v} вартість: C {delta(v_star['C'], v_std['C'])}; "
      f"FN {delta(v_star['FN'], v_std['FN'])}, FP {delta(v_star['FP'], v_std['FP'])}.")
    a(f"- **Test:** t* {word_t} вартість відносно 0.5: C {delta(t_star_['C'], t_std['C'])}; "
      f"FN {delta(t_star_['FN'], t_std['FN'])}, FP {delta(t_star_['FP'], t_std['FP'])}.")
    gain_v = v_std["C"] - v_star["C"]
    gain_t = t_std["C"] - t_star_["C"]
    a(f"- Виграш у C: validation {gain_v:+d}, test {gain_t:+d}. "
      f"На validation виграш невід'ємний за побудовою (поріг 0.5 теж еквівалентний одному з кандидатів, а t* - мінімум "
      f"серед усіх); на test знак і розмір виграшу не гарантовані.")
    a(f"- Позитивних у validation: {p_val}; у test: {te['star']['TP'] + te['star']['FN']}.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", type=Path, required=True,
                    help="Тека зі збереженими прогнозами (predictions.npz), напр. results")
    ap.add_argument("--out", type=Path, default=None,
                    help="Тека для результатів аналізу (за замовчуванням <saved>/analysis)")
    ap.add_argument("--readme", type=Path, default=HERE / "README.md",
                    help="README, у якому оновити блоки між маркерами (за замовчуванням README.md поруч)")
    ap.add_argument("--no-readme", action="store_true", help="Не змінювати README")
    args = ap.parse_args()
    out = args.out or args.saved / "analysis"
    run(args.saved, out, None if args.no_readme else args.readme)


if __name__ == "__main__":
    main()
