import os

from datasets import Dataset
import kagglehub
import pandas as pd


def load_and_process_kaggle_dataset():
    """
    Downloads the dataset from Kaggle, loads CSVs, processes them,
    and returns a single Hugging Face Dataset object.
    
    Returns:
        datasets.Dataset: A combined dataset with 'title', 'text', and 'label' columns.
    """
    # Krok 1: Pobierz zbiór danych
    print("Downloading 'fake-and-real-news-dataset' from Kaggle Hub...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    print(f"Dataset downloaded to: {path}")

    # Krok 2: Zdefiniuj ścieżki i wczytaj pliki
    fake_news_path = os.path.join(path, "Fake.csv")
    true_news_path = os.path.join(path, "True.csv")
    
    print("Loading CSV files into pandas DataFrames...")
    df_fake = pd.read_csv(fake_news_path)
    df_true = pd.read_csv(true_news_path)

    # Krok 3: Dodaj etykiety (1 dla fake, 0 dla true)
    df_fake['label'] = 1
    df_true['label'] = 0

    # Krok 4: Połącz ramki danych i wybierz odpowiednie kolumny
    print("Combining datasets...")
    combined_df = pd.concat([df_fake, df_true], ignore_index=True)
    final_df = combined_df[['title', 'text', 'label']]

    # Krok 5: Połącz kolumny tekstowe i usuń oryginalne
    print("Combining 'title' and 'text' into 'full_text'...")
    final_df['full_text'] = final_df['title'] + " " + final_df['text']
    final_df = final_df.drop(columns=['title', 'text'])

    # Krok 6: Konwertuj pandas DataFrame na Hugging Face Dataset
    print("Converting pandas DataFrame to Hugging Face Dataset...")
    hf_dataset = Dataset.from_pandas(final_df)
    
    return hf_dataset
