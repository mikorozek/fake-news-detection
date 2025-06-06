from datasets import load_dataset, concatenate_datasets

from src.config import Config


def load_datasets_and_concat(split: str):
    datasets = [load_dataset(dataset_name) for dataset_name in Config.DATASET_NAMES]

    dss = [dataset[split].select_columns(Config.REQUIRED_COLUMNS) for dataset in datasets]

    combined_datasets = concatenate_datasets(dss)

    return combined_datasets
