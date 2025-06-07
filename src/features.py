from typing import Tuple
from datasets import load_dataset, concatenate_datasets
import os
import numpy as np
from datasets import load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import Config


def load_datasets_and_concat(split: str):
    datasets = [load_dataset(dataset_name) for dataset_name in Config.DATASET_NAMES]

    dss = [dataset[split].select_columns(Config.REQUIRED_COLUMNS) for dataset in datasets]

    combined_datasets = concatenate_datasets(dss)

    return combined_datasets


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def tokenize_and_save_dataset(split: str, tokenize_function):
    dataset = load_datasets_and_concat(split)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset.save_to_disk(Config.DATASET_PATHS[split])
    return dataset


def load_or_create_dataset(tokenizer: PreTrainedTokenizer, split: str):
    def tokenize_function(examples):
        return tokenizer(
            examples["title"],
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
        )

    if os.path.exists(Config.DATASET_PATHS[split]):
        try:
            dataset = load_from_disk(Config.DATASET_PATHS[split])
        except Exception as e:
            print(f"Error while reading {split} dataset: {e}")
            print(f"Beggining tokenization of {split} dataset")
            dataset = tokenize_and_save_dataset(split, tokenize_function)
    else:
        print(f"Tokenized {split} dataset does not exist. Beggining tokenization")
        dataset = tokenize_and_save_dataset(split, tokenize_function)

    return dataset
