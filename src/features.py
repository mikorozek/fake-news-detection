from datasets import load_dataset, concatenate_datasets


def load_and_combine_datasets(split: str):
    dataset1 = load_dataset("GonzaloA/fake_news")
    dataset2 = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

    required_columns = ["title", "text", "label"]

    ds1 = dataset1[split].select_columns(required_columns)
    ds2 = dataset2[split].select_columns(required_columns)
    combined_dataset = concatenate_datasets([ds1, ds2])

    return combined_dataset
