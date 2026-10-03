import sys

from run_all import RESULTS, all_passed, format_text, run

if __name__ == "__main__":
    ok = run(divide_by_n=True)
    bug = run(divide_by_n=False)
    detected = not all_passed(bug)
    same_loss = ok["loss_numpy"] == bug["loss_numpy"]
    text = "\n\n".join([
        "Правильна реалізація",
        format_text(ok),
        "Без ділення на N",
        format_text(bug),
        f"Втрата не змінилась: {"так" if same_loss else "ні"}\n"
        f"Помилку виявлено: {'так' if detected else 'ні'}",
    ])
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "experiment_bug.txt").write_text(text + "\n", encoding="utf-8")
    sys.exit(0 if (detected and same_loss) else 1)
