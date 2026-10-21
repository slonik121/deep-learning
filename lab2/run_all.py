"""Зручний запуск усього конвеєра ЛР2.

    # навчання (один раз) + аналіз:
    uv run python run_all.py --data /шлях/до/creditcard.csv --results results

    # лише аналіз зі збережених прогнозів (без навчання й без CSV):
    uv run python run_all.py --results results

Навчання виконує вихідний lab2_starter.py без змін; аналіз - analyze.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import analyze

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="Шлях до локального creditcard.csv. Якщо не вказано - лише аналіз.")
    ap.add_argument("--results", type=Path, default=HERE / "results",
                    help="Тека з результатами навчання (за замовчуванням ./results)")
    args = ap.parse_args()

    if args.data is not None:
        # Стартовий скрипт сам відмовиться працювати з непорожньою текою --out.
        subprocess.run(
            [sys.executable, str(HERE / "lab2_starter.py"), "--data", str(args.data), "--out", str(args.results)],
            check=True,
        )
    analyze.run(args.results, args.results / "analysis", HERE / "README.md")


if __name__ == "__main__":
    main()
