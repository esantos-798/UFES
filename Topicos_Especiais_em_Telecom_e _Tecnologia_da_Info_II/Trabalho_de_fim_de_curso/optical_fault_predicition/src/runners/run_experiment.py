import torch
import torch.optim as optim

from src.data.dataloader import get_dataloader
from src.models.model_factory import ModelFactory

from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

from src.evaluation.anomaly_detector import AnomalyDetector

from src.utils.experiment_logger import ExperimentLogger


class ExperimentRunner:

    def __init__(self, experiment):

        self.experiment = experiment

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.logger = ExperimentLogger(experiment)


    def run(self):

        print()
        print("=" * 60)
        print(self.experiment.name)
        print("=" * 60)
        print()

        # ======================================================
        # DATA
        # ======================================================

        train_loader, val_loader, test_loader = get_dataloader(

            batch_size=self.experiment.batch_size,

            task=self.experiment.task

        )

        # ======================================================
        # MODEL
        # ======================================================

        model = ModelFactory.create(

            self.experiment

        ).to(self.device)

        print(model)

        # ======================================================
        # OPTIMIZER
        # ======================================================

        optimizer = optim.Adam(

            model.parameters(),

            lr=self.experiment.lr

        )

        # ======================================================
        # TRAIN
        # ======================================================

        trainer = Trainer(

            experiment=self.experiment,

            model=model,

            train_loader=train_loader,

            val_loader=val_loader,

            optimizer=optimizer,

            criterion=self.experiment.criterion,

            device=self.device

        )

        model = trainer.fit()

        history = trainer.history

        # ======================================================
        # TEST
        # ======================================================

        if self.experiment.task == "classification":

            evaluator = Evaluator(

                model=model,

                test_loader=test_loader,

                device=self.device,

                task="classification"

            )

            results = evaluator.evaluate()

            predictions = evaluator.predictions

            labels = evaluator.labels

            errors = None

            threshold = None

        elif self.experiment.task == "forecast":

            detector = AnomalyDetector(

                model=model,

                test_loader=test_loader,

                device=self.device

            )

            results = detector.evaluate()

            predictions = detector.predictions

            labels = detector.labels

            errors = detector.errors

            threshold = detector.threshold

        else:

            raise ValueError("Unknown task")

        # ======================================================
        # PRINT
        # ======================================================

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        for k, v in results.items():

            print(f"{k}: {v}")

        # ======================================================
        # SAVE EVERYTHING
        # ======================================================

        self.logger.save_all(

            model=model,

            metrics=results,

            history=history,

            predictions=predictions,

            labels=labels,

            errors=errors,

            threshold=threshold

        )

        print()

        print("Experiment saved successfully.")

        return results