import os
import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

class Config:
    """Prosta klasa do przechowywania zmiennych konfiguracyjnych."""
    HF_DATASET_NAME = "allegro/klej-polemo2-in"
    PROCESSED_DATA_DIR = "processed_data_polemo2"
    DATASET_PATHS = {
        "train": os.path.join(PROCESSED_DATA_DIR, "train"),
        "validation": os.path.join(PROCESSED_DATA_DIR, "validation"),
        "test": os.path.join(PROCESSED_DATA_DIR, "test"),
    }

def load_and_process_split(split: str):
    """
    Pobiera i przetwarza określony podział danych (train/validation/test)
    z repozytorium na Hugging Face.
    """
    print(f"Pobieranie surowych danych '{split}' ze źródła {Config.HF_DATASET_NAME}...")
    dataset = load_dataset(Config.HF_DATASET_NAME, split=split)
    processed_dataset = dataset.rename_column("target", "label")
    print(f"Zakończono przetwarzanie danych '{split}'.")
    return processed_dataset

def get_or_create_all_splits():
    """
    Wczytuje przetworzone zbiory danych z dysku. Jeśli nie istnieją,
    tworzy je i zapisuje do przyszłego użytku.
    """
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    datasets = {}
    for split in ["train", "validation", "test"]:
        path = Config.DATASET_PATHS[split]
        if os.path.exists(path):
            print(f"Wczytywanie przetworzonego zbioru '{split}' z '{path}'...")
            datasets[split] = load_from_disk(path)
        else:
            print(f"Nie znaleziono przetworzonego zbioru '{split}'. Tworzenie i zapisywanie...")
            dataset = load_and_process_split(split)
            dataset.save_to_disk(path)
            datasets[split] = dataset
            print(f"Zbiór danych '{split}' zapisany w '{path}'.")
    return datasets["train"], datasets["validation"], datasets["test"]

def evaluate_model(pipeline, X_data, y_data, split_name: str):
    """Ocenia model na podanym zbiorze danych i wyświetla wyniki."""
    print(f"\n--- Wyniki ewaluacji dla zbioru: {split_name.upper()} ---")
    predictions = pipeline.predict(X_data)
    
    accuracy = accuracy_score(y_data, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_data, predictions, average="weighted", zero_division=0
    )
    
    print(f"Dokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision): {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"Wynik F1 (F1-Score): {f1:.4f}")
    print("------------------------------------------")

def main():
    """Główna funkcja skryptu."""
    print("--- Uruchamianie potoku SVM dla zbioru polemo2 ---")
    train_dataset, val_dataset, test_dataset = get_or_create_all_splits()

    print("\nPrzygotowywanie danych do treningu i ewaluacji...")
    X_train = train_dataset["sentence"]
    y_train = train_dataset["label"]
    
    X_val = val_dataset["sentence"]
    y_val = val_dataset["label"]

    X_test = test_dataset["sentence"]
    y_test = test_dataset["label"]

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(max_features=10000)),
            ("clf", LinearSVC(C=1.0, random_state=42, dual='auto', max_iter=2000)),
        ]
    )

    print("\nRozpoczynanie treningu modelu...")
    pipeline.fit(X_train, y_train)
    print("Trening zakończony.")

    # Ewaluacja na zbiorze walidacyjnym
    evaluate_model(pipeline, X_val, y_val, "walidacyjny")
    
    # Ewaluacja na zbiorze testowym
    evaluate_model(pipeline, X_test, y_test, "testowy")

    print("\n\n--- Testowanie przykładowej predykcji ---")
    label_map = {
        0: "Opinia niejednoznaczna (meta_amb)",
        1: "Opinia negatywna (meta_minus_m)",
        2: "Opinia pozytywna (meta_plus_m)",
        3: "Opinia neutralna (meta_zero)",
    }
    
    sample_text = "Ten telefon robi naprawdę świetne zdjęcia, jestem zachwycony."
    prediction = pipeline.predict([sample_text])

    print(f"Tekst: '{sample_text}'")
    print(f"Predykcja: {label_map.get(prediction[0], 'Nieznana etykieta')}")


if __name__ == "__main__":
    main()
