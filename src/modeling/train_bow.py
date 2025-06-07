import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class Config:
    """Prosta klasa do przechowywania zmiennych konfiguracyjnych."""
    PROCESSED_DATA_DIR = "processed_data_svm_combined"
    DATASET_PATHS = {
        "train": os.path.join(PROCESSED_DATA_DIR, "train"),
        "validation": os.path.join(PROCESSED_DATA_DIR, "validation"),
    }

def load_and_combine_raw_datasets(split: str):
    """
    Pobiera, łączy i przetwarza dwa zbiory danych z Hugging Face Hub.
    Ta funkcja jest wywoływana tylko wtedy, gdy przetworzona wersja nie zostanie znaleziona na dysku.
    """
    # Te zbiory nie mają podziału 'validation', więc używamy 'test' w tym przypadku.
    # To mapuje nasze żądanie 'validation' na istniejący podział 'test'.
    effective_split = "test" if split == "validation" else split

    print(f"Pobieranie surowych danych '{effective_split}' ze źródeł...")
    
    # Ładowanie dwóch zbiorów danych
    dataset1 = load_dataset("GonzaloA/fake_news", split=effective_split)
    dataset2 = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English", split=effective_split)

    # Wybieramy tylko potrzebne kolumny
    required_columns = ["title", "text", "label"]
    ds1 = dataset1.select_columns(required_columns)
    ds2 = dataset2.select_columns(required_columns)
    
    # Łączymy oba zbiory w jeden
    combined_dataset = concatenate_datasets([ds1, ds2])

    # Funkcja do połączenia kolumn 'title' i 'text' w jedną
    def combine_text_columns(example):
        example["full_text"] = str(example["title"]) + " " + str(example["text"])
        return example

    # Stosujemy funkcję do całego zbioru i usuwamy oryginalne kolumny tekstowe
    processed_dataset = combined_dataset.map(combine_text_columns, remove_columns=['title', 'text'])
    
    print(f"Zakończono przetwarzanie surowych danych dla podziału '{split}'.")
    return processed_dataset

def get_or_create_processed_datasets():
    """
    Wczytuje przetworzone zbiory danych z dysku, jeśli istnieją.
    Jeśli nie, wywołuje funkcję do ich utworzenia i zapisuje je do przyszłego użytku.
    """
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    
    if os.path.exists(Config.DATASET_PATHS["train"]):
        print(f"Wczytywanie przetworzonego zbioru treningowego z '{Config.DATASET_PATHS['train']}'...")
        train_dataset = load_from_disk(Config.DATASET_PATHS["train"])
    else:
        print("Nie znaleziono przetworzonego zbioru treningowego. Tworzenie i zapisywanie...")
        train_dataset = load_and_combine_raw_datasets("train")
        train_dataset.save_to_disk(Config.DATASET_PATHS["train"])
        print(f"Zbiór treningowy zapisany w '{Config.DATASET_PATHS['train']}'.")

    if os.path.exists(Config.DATASET_PATHS["validation"]):
        print(f"Wczytywanie przetworzonego zbioru walidacyjnego z '{Config.DATASET_PATHS['validation']}'...")
        val_dataset = load_from_disk(Config.DATASET_PATHS["validation"])
    else:
        print("Nie znaleziono przetworzonego zbioru walidacyjnego. Tworzenie i zapisywanie...")
        val_dataset = load_and_combine_raw_datasets("validation")
        val_dataset.save_to_disk(Config.DATASET_PATHS["validation"])
        print(f"Zbiór walidacyjny zapisany w '{Config.DATASET_PATHS['validation']}'.")

    return train_dataset, val_dataset


def main():
    print("--- Uruchamianie potoku SVM z Bag-of-Words na połączonych zbiorach ---")
    train_dataset, val_dataset = get_or_create_processed_datasets()

    print("\nPrzygotowywanie danych do treningu...")
    X_train = train_dataset["full_text"]
    y_train = train_dataset["label"]
    X_val = val_dataset["full_text"]
    y_val = val_dataset["label"]

    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
            ),
            (
                "clf",
                LinearSVC(C=1.0, random_state=42, dual='auto', max_iter=2000)
            ),
        ]
    )

    print("Rozpoczynanie treningu modelu...")
    pipeline.fit(X_train, y_train)
    print("Trening zakończony.")

    print("\nEwaluacja modelu na zbiorze walidacyjnym (testowym)...")
    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, predictions, average="weighted"
    )

    # Wyświetlanie wyników
    print("\n--- Wyniki ewaluacji ---")
    print(f"Dokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision): {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"Wynik F1 (F1-Score): {f1:.4f}")
    print("--------------------------\n")
    
    # Przykładowa predykcja na nowym tekście
    print("Testowanie przykładowej predykcji:")
    sample_text = "NASA confirms plans to build a permanent base on the Moon by 2030."
    prediction = pipeline.predict([sample_text])
    
    # Na podstawie analizy zbiorów: 0 = Fałszywy (Fake), 1 = Prawdziwy (Real)
    label_map = {0: "Fake News", 1: "Real News"}
    print(f"Tekst: '{sample_text}'")
    print(f"Predykcja: {label_map.get(prediction[0], 'Nieznana etykieta')}")


if __name__ == "__main__":
    main()
