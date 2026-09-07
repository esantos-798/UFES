import torch
from src.training.history import HistoryLogger

class ForecastTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        experiment,
        patience=5
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.experiment = experiment

        self.best_loss = float("inf")

        self.counter = 0

        self.patience = patience

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }

    # -----------------------------------

    def train_epoch(self):

        self.model.train()

        running_loss = 0

        for X, y, _ in self.train_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(X)

            loss = self.criterion(
                pred,
                y
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    # -----------------------------------

    def validate(self):

        self.model.eval()

        running_loss = 0

        with torch.no_grad():

            for X, y, _ in self.val_loader:

                X = X.to(self.device)

                y = y.to(self.device)

                pred = self.model(X)

                loss = self.criterion(
                    pred,
                    y
                )

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    # -----------------------------------

    def save_history(
        self,
        epoch,
        train_loss,
        val_loss
    ):

        self.history["epoch"].append(epoch)

        self.history["train_loss"].append(train_loss)

        self.history["val_loss"].append(val_loss)

    # -----------------------------------

    def early_stopping(self, val_loss):

        if val_loss < self.best_loss:

            print(
                f"Validation improved "
                f"{self.best_loss:.6f} -> {val_loss:.6f}"
            )

            self.best_loss = val_loss

            self.counter = 0

            torch.save(
                self.model.state_dict(),
                self.experiment.checkpoint
            )

            self.experiment.save_history(
                self.history
            )

            return False

        self.counter += 1

        print(
            f"No improvement "
            f"({self.counter}/{self.patience})"
        )

        return self.counter >= self.patience