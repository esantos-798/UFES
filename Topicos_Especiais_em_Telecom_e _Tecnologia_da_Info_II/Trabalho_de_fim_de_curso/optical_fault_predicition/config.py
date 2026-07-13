"""
config.py

Centraliza todas as configurações do projeto.

Qualquer hiperparâmetro deve ser alterado apenas aqui.
"""

from pathlib import Path
import torch

# ==========================================================
# Diretórios
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "datasets"

OUTPUT_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"

# Cria os diretórios automaticamente
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
METRIC_DIR.mkdir(exist_ok=True)

# ==========================================================
# Arquivos dos datasets
# ==========================================================

HARD_FAILURE_DATASET = DATASET_DIR / "HardFailure_dataset.csv"
SOFT_FAILURE_DATASET = DATASET_DIR / "SoftFailure_dataset.csv"

# ==========================================================
# Colunas
# ==========================================================

TIMESTAMP_COLUMN = "Timestamp"

ID_COLUMN = "ID"

TARGET_COLUMN = "Failure"

FEATURE_COLUMNS = [
    "BER",
    "OSNR",
    "InputPower",
    "OutputPower"
]

# ==========================================================
# Pré-processamento
# ==========================================================

NORMALIZATION = "minmax"
# opções:
#   "minmax"
#   "standard"

WINDOW_SIZE = 30

PREDICTION_HORIZON = 1

TRAIN_RATIO = 0.80

SHUFFLE_TRAIN = False

# ==========================================================
# Treinamento
# ==========================================================

BATCH_SIZE = 64

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

RANDOM_SEED = 42

# ==========================================================
# Modelos
# ==========================================================

INPUT_SIZE = len(FEATURE_COLUMNS)

HIDDEN_SIZE = 100

NUM_LAYERS = 1

DROPOUT = 0.2

INPUT_SIZE = 12

OUTPUT_SIZE = 1

HIDDEN_SIZE = 100

# ==========================================================
# Hardware
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Reprodutibilidade
# ==========================================================

PIN_MEMORY = True

NUM_WORKERS = 0

# ======================================
# Training
# ======================================

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

EPOCHS = 30

PATIENCE = 5
