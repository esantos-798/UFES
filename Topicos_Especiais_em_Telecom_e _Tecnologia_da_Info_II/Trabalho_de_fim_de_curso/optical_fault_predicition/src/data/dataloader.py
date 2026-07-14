import numpy as np

from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from .dataset import OpticalDataset
from .preprocessing import OpticalFaultPreprocessor

#from config import SOFT_FAILURE_DATASET

from config import HARD_FAILURE_DATASET



def get_dataloader(
    batch_size=64,
    shuffle=True,
    task="classification",
    dataset_path=HARD_FAILURE_DATASET
    #dataset_path=SOFT_FAILURE_DATASET
):

    # ============================
    # Preprocessamento
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
    # Split aleatório estratificado
    # Reprodução do artigo
    # ============================

    total_size = len(dataset)

    indices = np.arange(
        total_size
    )


    if task == "classification":

        labels = failure

    else:

        labels = None



    # ============================
    # Train / Test
    # ============================

    if task == "classification":

        train_indices, test_indices = train_test_split(

            indices,

            test_size=0.15,

            random_state=42,

            stratify=labels

        )

    else:

        train_indices, test_indices = train_test_split(

            indices,

            test_size=0.15,

            random_state=42

        )



    # ============================
    # Train / Validation
    # ============================

    if task == "classification":

        train_indices, val_indices = train_test_split(

            train_indices,

            test_size=0.1765,

            random_state=42,

            stratify=labels[train_indices]

        )

    else:

        train_indices, val_indices = train_test_split(

            train_indices,

            test_size=0.1765,

            random_state=42

        )



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
        "\nRandom stratified split:"
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