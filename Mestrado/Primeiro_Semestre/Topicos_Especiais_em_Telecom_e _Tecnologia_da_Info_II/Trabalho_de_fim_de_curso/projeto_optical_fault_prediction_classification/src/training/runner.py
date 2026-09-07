import os
import json
import torch

from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

from src.utils.plots import (
    plot_losses,
    plot_roc_curve
)


class ExperimentRunner:

    def __init__(
        self,
        experiment_name,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=30,
        patience=5
    ):

        self.experiment_name = experiment_name

        self.save_dir = os.path.join(
            "experiments",
            experiment_name
        )

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        self.model = model

        self.device = device

        self.epochs = epochs

        self.trainer = Trainer(

            model=model,

            train_loader=train_loader,

            val_loader=val_loader,

            optimizer=optimizer,

            criterion=criterion,

            device=device,

            patience=patience,

            checkpoint_path=os.path.join(
                self.save_dir,
                "best_model.pt"
            )
        )

        self.test_loader = test_loader

    def run(self):

        print(f"\n========== {self.experiment_name.upper()} ==========\n")

        for epoch in range(self.epochs):

            train_loss = self.trainer.train_epoch()

            val_loss = self.trainer.validate()

            self.trainer.train_losses.append(train_loss)

            self.trainer.val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1:02d}/{self.epochs}"
                f" | Train {train_loss:.5f}"
                f" | Val {val_loss:.5f}"
            )

            if self.trainer.early_stopping(val_loss):

                print("Early stopping")

                break

        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    self.save_dir,
                    "best_model.pt"
                )
            )
        )

        evaluator = Evaluator(
            self.model,
            self.test_loader,
            self.device
        )

        results = evaluator.evaluate()

        with open(

            os.path.join(
                self.save_dir,
                "metrics.json"
            ),

            "w"

        ) as f:

            json.dump(

                {

                    "Accuracy": float(results["Accuracy"]),

                    "Precision": float(results["Precision"]),

                    "Recall": float(results["Recall"]),

                    "F1": float(results["F1"]),

                    "AUC": float(results["AUC"])

                },

                f,

                indent=4

            )

        plot_losses(

            self.trainer.train_losses,

            self.trainer.val_losses,

            save_path=os.path.join(
                self.save_dir,
                "loss.png"
            )

        )

        plot_roc_curve(

            results["FPR"],

            results["TPR"],

            results["AUC"],

            save_path=os.path.join(
                self.save_dir,
                "roc.png"
            )

        )

        print("\nResults")

        for k, v in results.items():

            if k in ["FPR", "TPR", "ConfusionMatrix"]:

                continue

            print(f"{k}: {v}")

        return results