from pathlib import Path


# ==================================================
# Project paths
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = ROOT_DIR / "data"

RESULTS_DIR = ROOT_DIR / "results"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


# Criar diretórios automaticamente

RESULTS_DIR.mkdir(
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# Dataset configuration
# ==================================================

DATASETS = {

    "hard_failure":
        DATA_DIR / "HardFailure_dataset.csv",

    "soft_failure":
        DATA_DIR / "SoftFailure_dataset.csv"

}


# ==================================================
# Default experiment parameters
# ==================================================

DEFAULT_CONFIG = {


    # Data

    "dataset": "hard_failure",

    "task": "forecast",


    # Sequence

    "window_size": 30,

    "input_size": 12,

    "output_size": 12,


    # Training

    "batch_size": 64,

    "epochs": 30,

    "learning_rate": 1e-3,


    # Split

    "train_ratio": 0.70,

    "val_ratio": 0.15,

    "test_ratio": 0.15,


    # Model

    "hidden_size": 100,

    "dropout": 0.2,


    # Reproducibility

    "seed": 42

}