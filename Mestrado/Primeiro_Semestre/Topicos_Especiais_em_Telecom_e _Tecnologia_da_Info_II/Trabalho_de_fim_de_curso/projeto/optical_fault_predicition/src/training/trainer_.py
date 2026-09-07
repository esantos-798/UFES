import copy
import torch
from torch import autocast
from torch.amp import GradScaler

class Trainer:

    def __init__(
        self,
        experiment,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        scheduler=None
    ):
        self.experiment = experiment
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.epochs = experiment.epochs
        self.patience = experiment.patience  # Lido com segurança do Config centralizado

        self.best_loss = float("inf")
        self.best_state = None
        self.wait = 0
        self.clip_grad = 1.0
        self.use_amp = torch.cuda.is_available()

        self.scaler = GradScaler(
            device="cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_amp
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": None,
        }

    def normalize_prediction(self, prediction):
        if isinstance(prediction, torch.Tensor):
            return {"forecast": prediction}
        if isinstance(prediction, tuple):
            result = {}
            if len(prediction) >= 1:
                result["forecast"] = prediction[0]
            if len(prediction) >= 2:
                result["failure"] = prediction[1]
            return result
        if isinstance(prediction, dict):
            return prediction
        raise TypeError(f"Unsupported model output type: {type(prediction)}")
    
    def compute_loss(self, prediction, target, failure=None):
        # Utiliza o método de normalização que você já criou para extrair um dicionário limpo
        preds_dict = self.normalize_prediction(prediction)
        
        forecast_pred = preds_dict.get("forecast")
        failure_pred = preds_dict.get("failure", None)

        # Se por acaso o forecast ainda vier envelopado em tupla/lista por resíduo do forward
        if isinstance(forecast_pred, (tuple, list)):
            forecast_pred = forecast_pred[0]
            
        if isinstance(failure_pred, (tuple, list)):
            failure_pred = failure_pred[0]

        # Calcula a perda de forecast usando apenas o tensor correto
        forecast_loss = self.criterion(forecast_pred, target)
        
        # Lógica de perda combinada (Multi-Task):
        # Como o Trainer não recebe a 'failure_loss' no __init__, vamos buscá-la dinamicamente no experimento
        failure_criterion = getattr(self.experiment, 'failure_loss', None)
        
        if failure_pred is not None and failure is not None and failure_criterion is not None:
            # Garante que as dimensões do target de falha batam com a saída do modelo
            if failure_pred.dim() != failure.dim():
                target_fail = failure.float().unsqueeze(-1) if failure_pred.dim() > failure.dim() else failure.float().squeeze(-1)
            else:
                target_fail = failure.float()
                
            f_loss = failure_criterion(failure_pred, target_fail)
            
            # Recupera os pesos do Grid Search com fallback seguro (= 1.0)
            f_weight = getattr(self.experiment, 'forecast_weight', 1.0)
            fail_weight = getattr(self.experiment, 'failure_weight', 1.0)
            
            return (f_weight * forecast_loss) + (fail_weight * f_loss)
            
        return forecast_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for X, y, failure in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            failure = failure.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            with autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.use_amp
            ):
                prediction = self.model(X)
                loss = self.compute_loss(prediction, y, failure)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for X, y, failure in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                failure = failure.to(self.device).unsqueeze(1)

                prediction = self.model(X)
                loss = self.compute_loss(prediction, y, failure)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def update_best_model(self, val_loss, epoch):
        if val_loss < self.best_loss:
            print(f"Validation improved {self.best_loss:.6f} -> {val_loss:.6f}")
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.history["best_epoch"] = epoch + 1
            self.wait = 0
        else:
            self.wait += 1
            print(f"No improvement ({self.wait}/{self.patience})")

        return self.wait >= self.patience

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch+1:02d}/{self.epochs} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f}"
            )

            stop = self.update_best_model(val_loss, epoch)

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            if stop:
                print("\nEarly stopping")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print()
        print("Best validation loss:", f"{self.best_loss:.6f}")
        print("Best epoch:", self.history["best_epoch"])

        return self.model

    def get_history(self):
        return self.history