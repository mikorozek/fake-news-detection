from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import load_datasets_and_concat


def evaluate_model(pipeline, X_data, y_data, split_name: str):
    print(f"\n--- Evaluation score for split: {split_name.upper()} ---")
    predictions = pipeline.predict(X_data)

    accuracy = accuracy_score(y_data, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_data, predictions, average="weighted"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def main():
    train_dataset = load_datasets_and_concat("train")
    train_dataset = train_dataset.rename_column("target", "label")
    val_dataset = load_datasets_and_concat("val")
    val_dataset = val_dataset.rename_column("target", "label")
    test_dataset = load_datasets_and_concat("test")
    test_dataset = test_dataset.rename_column("target", "label")

    X_train = train_dataset["sentence"]
    y_train = train_dataset["label"]

    X_val = val_dataset["sentence"]
    y_val = val_dataset["label"]

    X_test = test_dataset["sentence"]
    y_test = test_dataset["label"]

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(max_features=10000)),
            ("clf", LinearSVC(C=1.0, random_state=42, dual="auto", max_iter=2000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    print("Training finished")

    evaluate_model(pipeline, X_val, y_val, "val")

    evaluate_model(pipeline, X_test, y_test, "test")

    label_map = {
        0: "(meta_amb)",
        1: "(meta_minus_m)",
        2: "(meta_plus_m)",
        3: "(meta_zero)",
    }

    sample_text = "Ten telefon robi naprawdę świetne zdjęcia, jestem zachwycony."
    prediction = pipeline.predict([sample_text])

    print(f"Text: '{sample_text}'")
    print(f"Prediction: {label_map.get(prediction[0], 'Unknown label')}")


if __name__ == "__main__":
    main()
