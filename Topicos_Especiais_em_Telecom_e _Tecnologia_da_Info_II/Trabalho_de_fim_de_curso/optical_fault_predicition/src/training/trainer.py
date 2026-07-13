import torch

from src.training.history import History


class Trainer:

    def __init__(

        self,

        experiment,

        model,

        train_loader,

        val_loader,

        optimizer,

        criterion,

        device

    ):

        self.experiment = experiment

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.history = History(experiment)

        self.best_loss = float("inf")

        self.counter = 0

    # =======================================================
    # Train one epoch
    # =======================================================

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for batch in self.train_loader:

            x = batch[0].to(self.device)

            y = batch[1].to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(x)

            loss = self.criterion(

                prediction,

                y

            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # =======================================================
    # Validation
    # =======================================================

    def validate(self):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for batch in self.val_loader:

                x = batch[0].to(self.device)

                y = batch[1].to(self.device)

                prediction = self.model(x)

                loss = self.criterion(

                    prediction,

                    y

                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # =======================================================
    # Early stopping
    # =======================================================

    def early_stopping(

        self,

        val_loss

    ):

        if val_loss < self.best_loss:

            print(

                f"Validation improved "

                f"{self.best_loss:.6f} -> "

                f"{val_loss:.6f}"

            )

            self.best_loss = val_loss

            self.counter = 0

            torch.save(

                self.model.state_dict(),

                self.experiment.checkpoint

            )

            return False

        self.counter += 1

        print(

            f"No improvement "

            f"({self.counter}/"

            f"{self.experiment.patience})"

        )

        return self.counter >= self.experiment.patience

    # =======================================================
    # Main training loop
    # =======================================================

    def fit(self):

        print()

        print("=" * 60)

        print(self.experiment.name)

        print("=" * 60)

        print()

        for epoch in range(self.experiment.epochs):

            self.history.start_epoch()

            train_loss = self.train_epoch()

            val_loss = self.validate()

            lr = self.optimizer.param_groups[0]["lr"]

            self.history.end_epoch(

                epoch=epoch + 1,

                train_loss=train_loss,

                val_loss=val_loss,

                lr=lr

            )

            print(

                f"Epoch "

                f"{epoch+1:02d}/"

                f"{self.experiment.epochs:02d}"

                f" | "

                f"Train {train_loss:.6f}"

                f" | "

                f"Val {val_loss:.6f}"

            )

            stop = self.early_stopping(

                val_loss

            )

            if stop:

                print()

                print("Early stopping")

                break

        self.history.save_all()

        self.model.load_state_dict(

            torch.load(

                self.experiment.checkpoint,

                weights_only=True

            )

        )

        return self.model