from pathlib import Path
from src.config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from src.features import compute_metrics, load_or_create_dataset


def load_model_from_checkpoint(checkpoint_path, num_labels):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if not (checkpoint_path / "model.safetensors").exists():
        raise FileNotFoundError(f"No model safetensors in: {checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=num_labels
    )

    return model


def predict_batch(model, dataset, batch_size=32):
    model.eval()
    logits = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"]),
                "attention_mask": torch.tensor(batch["attention_mask"]),
            }
            outputs = model(**inputs)
            logits.extend(outputs.logits.tolist())

    return logits


def main():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    test_dataset = load_or_create_dataset(tokenizer, "test")
    num_labels = len(set(test_dataset["label"]))
    model = load_model_from_checkpoint(Config.MODEL_CHECKPOINT_PATH, num_labels)
    logits = predict_batch(model, test_dataset)
    metrics_dict = compute_metrics((logits, test_dataset["label"]))
    for metric, score in metrics_dict.items():
        print(f"{metric} score on test dataset: {score}")

    save_path = Path(Config.PLOTS_DIR) / "test_confusion_matrix.png"
    predictions = np.argmax(logits, axis=1)
    cm = confusion_matrix(test_dataset["label"], predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.savefig(str(save_path))
    plt.close()
