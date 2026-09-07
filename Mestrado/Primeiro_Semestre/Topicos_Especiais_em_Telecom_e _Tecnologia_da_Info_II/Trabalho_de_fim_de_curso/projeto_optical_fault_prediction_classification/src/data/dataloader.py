import numpy as np
from torch.utils.data import DataLoader, Subset

from .dataset import OpticalDataset
from .preprocessing import OpticalFaultPreprocessor

# Configurações de Dataset
from config import HARD_FAILURE_DATASET, SOFT_FAILURE_DATASET


def get_dataloader(
    batch_size=64,
    shuffle=True,
    task="forecast",  # Ajustado typo de "forescat" -> "forecast"
    dataset_path=HARD_FAILURE_DATASET
):

    # ============================
    # Pré-processamento
    # ============================
    processor = OpticalFaultPreprocessor(
        dataset_path
    )

    X, y, failure = processor.run()

    dataset = OpticalDataset(
        X,
        y,
        failure=failure,
        task=task
    )

    print("X:", X.shape)
    print("y:", y.shape)

    if task == "classification":
        print(
            "failure:",
            failure.shape
        )

    # ============================
    # Split Cronológico Sequencial
    # (Evita Data Leakage entre janelas)
    # ============================
    total_size = len(dataset)

    # Limites das fatias no tempo (70% Treino | 15% Validação | 15% Teste)
    train_end = int(total_size * 0.70)
    val_end = int(total_size * 0.85)

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, total_size)

    # Subsets baseados na cronologia dos dados
    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    test_dataset = Subset(
        dataset,
        test_indices
    )

    print(
        "\nSequential time-based split (No Data Leakage):"
    )

    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    print(
        "Test:",
        len(test_dataset)
    )

    # ============================
    # Distribuição de falhas
    # ============================
    if task == "classification":
        print(
            "\nFailure distribution:"
        )

        print(
            "Train:",
            count_failures(train_dataset)
        )

        print(
            "Validation:",
            count_failures(val_dataset)
        )

        print(
            "Test:",
            count_failures(test_dataset)
        )

    # ============================
    # DataLoaders
    # ============================
    # ATENÇÃO: shuffle=True no DataLoader de treino embaralha APENAS a ordem
    # das janelas durante o gradiente, sem misturar os conjuntos de validação e teste.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ==========================================================
# Contagem de classes
# ==========================================================
def count_failures(dataset):

    labels = []

    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(
            item[1].item()
        )

    return np.unique(
        labels,
        return_counts=True
    )