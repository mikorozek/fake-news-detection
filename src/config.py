class Config:
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_LENGTH = 4096

    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    WEIGHT_DECAY = 0.01

    OUTPUT_DIR = "./results"
    BEST_MODEL_PATH = "/home/mrozek/fake-news-detection/models"
    DATASET_PATHS = {
        "train": "/home/mrozek/fake-news-detection/data/train_tokenized",
        "validation": "/home/mrozek/fake-news-detection/data/val_tokenized",
    }

    WANDB_PROJECT = "fake-news-classification"
