import argparse
import json
import sys
from pathlib import Path

import test_stability
from checks import compare_with_torch, numeric_check
from data import load_split
from mlp_numpy import init_params

RESULTS = Path(__file__).parent / "results"


def run(divide_by_n: bool = True) -> dict:
    X, y, Xte, yte = load_split()
    assert X.shape == (105, 4) and Xte.shape == (45, 4)
    params = init_params(0)

    test_stability.run()
    loss_np, loss_t, torch_rows = compare_with_torch(params, X, y, divide_by_n)
    num_rows = numeric_check(params, X, y, divide_by_n)
    return {
        "divide_by_n": divide_by_n,
        "n_train": int(X.shape[0]),
        "loss_numpy": loss_np,
        "loss_torch": loss_t,
        "torch_rows": [dict(name=n, max_abs_diff=d, passed=p) for n, d, p in torch_rows],
        "numeric_rows": [dict(param=n, backward=g, numeric=gn, abs_diff=d, passed=p)
                         for n, g, gn, d, p in num_rows],
    }


def yn(p: bool) -> str:
    return "так" if p else "ні"


def format_text(res: dict) -> str:
    mode = "правильний backward" if res["divide_by_n"] else "без ділення на N"
    out = [f"Режим: {mode}",
           f"Втрата NumPy:   {res['loss_numpy']:.15f}",
           f"Втрата PyTorch: {res['loss_torch']:.15f}",
           "",
           "Порівняння NumPy / PyTorch (максимальна абсолютна різниця, перевірку пройдено):"]
    for r in res["torch_rows"]:
        out.append(f"{r['name']:<14}{r['max_abs_diff']:.3e}   {yn(r['passed'])}")
    out += ["",
            "Чисельна перевірка (параметр, backward, чисельна похідна, абсолютна різниця, пройдено):"]
    for r in res["numeric_rows"]:
        out.append(f"{r['param']:<10}{r['backward']:+.12e}   {r['numeric']:+.12e}   "
                   f"{r['abs_diff']:.3e}   {yn(r['passed'])}")
    return "\n".join(out)


def all_passed(res: dict) -> bool:
    return all(r["passed"] for r in res["torch_rows"]) and all(r["passed"] for r in res["numeric_rows"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", action="store_true", help="skip division by N in the logits gradient")
    args = ap.parse_args(argv)

    res = run(divide_by_n=not args.bug)
    text = format_text(res)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    tag = "bug" if args.bug else "correct"
    (RESULTS / f"results_{tag}.txt").write_text(text + "\n", encoding="utf-8")
    (RESULTS / f"results_{tag}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = all_passed(res)
    if args.bug:
        print("\nПомилку виявлено" if not ok else "\nПомилку не виявлено")
        return 0 if not ok else 1
    print("\nУсі перевірки пройдено" if ok else "\nПеревірки не пройдено")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
