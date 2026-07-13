import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model_factory import ModelFactory

from src.data.dataloader import get_dataloader

from src.training.forecast_trainer import ForecastTrainer
from src.training.forecast_evaluator import ForecastEvaluator


class ExperimentRunner:


    def __init__(self, experiment):

        self.experiment = experiment

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


    def run(self):

        print("\n==============================")
        print(
            f"Running model: {self.experiment.model}"
        )
        print("==============================\n")


        # =========================
        # Data
        # =========================

        train_loader, val_loader, test_loader = get_dataloader(
            task=self.experiment.task
        )


        # =========================
        # Model
        # =========================

        model = ModelFactory.create(
            self.experiment
        )


        model = model.to(
            self.device
        )


        print(model)


        # =========================
        # Loss
        # =========================

        criterion = nn.HuberLoss()


        # =========================
        # Optimizer
        # =========================

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.experiment.learning_rate
        )


        # =========================
        # Trainer
        # =========================

        trainer = ForecastTrainer(

            model=model,

            train_loader=train_loader,

            val_loader=val_loader,

            optimizer=optimizer,

            criterion=criterion,

            device=self.device,

            patience=5,

            checkpoint_path=
            f"best_{self.experiment.model}.pt"

        )


        # =========================
        # Training
        # =========================

        for epoch in range(
            self.experiment.epochs
        ):

            train_loss = trainer.train_epoch()

            val_loss = trainer.validate()


            print(
                f"Epoch {epoch+1:02d}/"
                f"{self.experiment.epochs} "
                f"| Train {train_loss:.6f} "
                f"| Val {val_loss:.6f}"
            )


            stop = trainer.early_stopping(
                val_loss
            )


            if stop:

                print(
                    "Early stopping"
                )

                break



        # =========================
        # Load best model
        # =========================

        model.load_state_dict(
            torch.load(
                f"best_{self.experiment.model}.pt",
                weights_only=True
            )
        )


        # =========================
        # Evaluation
        # =========================

        evaluator = ForecastEvaluator(

            model=model,

            test_loader=test_loader,

            device=self.device

        )


        results = evaluator.evaluate()


        print("\n===== RESULTS =====")

        for key,value in results.items():

            print(
                f"{key}: {value:.6f}"
            )


        return results