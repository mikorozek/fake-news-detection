import os
from datasets import load_from_disk
from torch.nn.modules import padding
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.features import load_and_combine_datasets
from src.config import Config


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def tokenize_and_save_dataset(split, tokenize_function):
    dataset = load_and_combine_datasets(split)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset.save_to_disk(Config.DATASET_PATHS[split])
    return dataset


def load_or_create_datasets(tokenizer: LongformerTokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["title"],
            examples["text"],
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        )

    if os.path.exists(Config.DATASET_PATHS["train"]):
        try:
            train_dataset = load_from_disk(Config.DATASET_PATHS["train"])
        except Exception as e:
            print(f"Error while reading train dataset: {e}")
            print(f"Beggining tokenization of train dataset")
            train_dataset = tokenize_and_save_dataset("train", tokenize_function)
    else:
        print(f"Tokenized train dataset does not exist. Beggining tokenization")
        train_dataset = tokenize_and_save_dataset("train", tokenize_function)

    if os.path.exists(Config.DATASET_PATHS["validation"]):
        try:
            val_dataset = load_from_disk(Config.DATASET_PATHS["validation"])
        except Exception as e:
            print(f"Error while reading validation dataset: {e}")
            print(f"Beggining tokenization of validation dataset")
            val_dataset = tokenize_and_save_dataset("validation", tokenize_function)
    else:
        print(f"Tokenized validation dataset does not exist. Beggining tokenization")
        val_dataset = tokenize_and_save_dataset("validation", tokenize_function)

    return train_dataset, val_dataset


def main():
    wandb.init(
        project=Config.WANDB_PROJECT,
        config={
            "model_name": Config.MODEL_NAME,
            "max_length": Config.MAX_LENGTH,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "num_epochs": Config.NUM_EPOCHS,
            "warmup_steps": Config.WARMUP_STEPS,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    )

    tokenizer = LongformerTokenizer.from_pretrained(Config.MODEL_NAME)
    model = LongformerForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)

    train_dataset, val_dataset = load_or_create_datasets(tokenizer)

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_steps=Config.SAVE_STEPS,
        logging_steps=100,
        report_to="wandb",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(Config.BEST_MODEL_PATH)

    wandb.finish()


if __name__ == "__main__":
    main()
