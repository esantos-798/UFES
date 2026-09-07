import matplotlib.pyplot as plt
from pathlib import Path


def plot_losses(
    train_losses,
    val_losses,
    save_path=None
):

    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    else:

        plt.show()

    plt.close()


def plot_roc_curve(
    fpr,
    tpr,
    auc_score,
    save_path=None
):

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc_score:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    else:

        plt.show()

    plt.close()


    def plot_prediction(
        target,
        prediction,
        samples=200
    ):

        plt.figure(figsize=(12,5))

        plt.plot(
            target[:samples],
            label="Real"
        )

        plt.plot(
            prediction[:samples],
            label="Prediction"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()