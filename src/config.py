class Config:
    MODEL_NAME = "BAAI/bge-m3"
    MAX_LENGTH = 8192

    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    WEIGHT_DECAY = 0.01

    OUTPUT_DIR = "results"
    PLOTS_DIR = "plots"
    BEST_MODEL_PATH = "/models"
    DATASET_NAMES = ["allegro/klej-polemo2-in"]
    DATASET_PATHS = {
        "train": "data/train_tokenized",
        "validation": "data/val_tokenized",
        "test": "data/test_tokenized",
    }
    MODEL_CHECKPOINT_PATH = "results/checkpoint-600"
    REQUIRED_COLUMNS = ["sentence, target"]

    WANDB_PROJECT = "sentiment-classification"
