import json
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config


def collect_training_data():
    results_path = Path(Config.OUTPUT_DIR)

    checkpoint_datas = []

    checkpoint_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            match = re.search(r"checkpoint-(\d+)", item.name)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoint_dirs.append((checkpoint_num, item))

    checkpoint_dirs.sort(key=lambda x: x[0])
    for checkpoint_num, checkpoint_dir in checkpoint_dirs:
        trainer_state_path = checkpoint_dir / "trainer_state.json"

        data = {}
        try:
            with open(trainer_state_path, "r") as file_handle:
                trainer_state = json.load(file_handle)
            log_history = trainer_state.get("log_history", [])
            train_losses = []
            for entry in log_history:
                if "loss" in entry:
                    train_losses.append(entry["loss"])
                if "eval_loss" in entry:
                    data["eval_loss"] = entry["eval_loss"]
                    data["eval_accuracy"] = entry["eval_accuracy"]
                    data["eval_precision"] = entry["eval_precision"]
                    data["eval_recall"] = entry["eval_recall"]
                    data["eval_f1"] = entry["eval_recall"]
                    data["epoch"] = entry["epoch"]
            data["train_loss"] = np.mean(train_losses)
        except Exception as e:
            print(f"ERROR reading {checkpoint_dir.name}: {e}")
        checkpoint_datas.append(data)

    checkpoint_datas.sort(key=lambda x: x["epoch"])
    return checkpoint_datas


def plot_metric(data, metric_name):
    epochs = [checkpoint_data["epoch"] for checkpoint_data in data]
    values = [checkpoint_data[metric_name] for checkpoint_data in data]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{metric_name.replace("_", " ").title()} over epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = Path(Config.PLOTS_DIR) / f"{metric_name}.png"
    plt.savefig(str(save_path))
    plt.close()


def plot_metrics_combined(data, metrics_to_plot, filename):
    epochs = [checkpoint_data["epoch"] for checkpoint_data in data]
    plt.figure(figsize=(10, 6))

    for metric in metrics_to_plot:
        metric_values = [checkpoint_data[metric] for checkpoint_data in data]
        plt.plot(epochs, metric_values, linewidth=2, label=metric)

    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = Path(Config.PLOTS_DIR) / filename
    plt.savefig(str(save_path))
    plt.close()


def main():
    data = collect_training_data()
    plot_metric(data, "train_loss")
    plot_metric(data, "eval_loss")
    plot_metric(data, "eval_accuracy")
    plot_metric(data, "eval_precision")
    plot_metric(data, "eval_recall")
    plot_metric(data, "eval_f1")
    metrics_to_plot = ["train_loss", "eval_loss"]
    plot_metrics_combined(data, metrics_to_plot, "metrics_combined.png")


if __name__ == "__main__":
    main()
