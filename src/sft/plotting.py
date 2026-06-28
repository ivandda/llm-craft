from __future__ import annotations

import json
from pathlib import Path


def read_loss_jsonl(path: str | Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    file_path = Path(path)
    if not file_path.exists():
        return steps, losses
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            steps.append(int(record["step"]))
            losses.append(float(record["loss"]))
    return steps, losses


def plot_losses(run_dir: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    run_path = Path(run_dir)
    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train_steps, train_losses = read_loss_jsonl(run_path / "train_losses.jsonl")
    dev_steps, dev_losses = read_loss_jsonl(run_path / "eval_losses.jsonl")

    if train_losses:
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.xlabel("step")
        plt.ylabel("train loss")
        plt.tight_layout()
        plt.savefig(plots_dir / "train_loss.png")
        plt.close()

    if dev_losses:
        plt.figure()
        plt.plot(dev_steps, dev_losses)
        plt.xlabel("step")
        plt.ylabel("dev loss")
        plt.tight_layout()
        plt.savefig(plots_dir / "dev_loss.png")
        plt.close()

    if train_losses or dev_losses:
        plt.figure()
        if train_losses:
            plt.plot(train_steps, train_losses, label="train")
        if dev_losses:
            plt.plot(dev_steps, dev_losses, label="dev")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "losses_combined.png")
        plt.close()
