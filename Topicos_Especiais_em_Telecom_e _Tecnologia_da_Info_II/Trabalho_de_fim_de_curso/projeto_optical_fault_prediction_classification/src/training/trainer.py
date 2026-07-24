import copy
import torch
# Atualização dos pacotes de precisão mista (AMP) para evitar FutureWarnings
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
        self.patience = experiment.patience

        self.best_loss = float("inf")
        self.best_state = None
        self.wait = 0

        self.clip_grad = 1.0

        self.use_amp = torch.cuda.is_available()

        # Correção do GradScaler para sintaxe estável do PyTorch 2.x
        self.scaler = GradScaler(
            device="cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_amp
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": None,
        }

    ##########################################################
    # NORMALIZA SAIDA DO MODELO
    ##########################################################

    def normalize_prediction(self, prediction):
        """
        Padroniza saída dos modelos para formato dict.
        """
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

        raise TypeError(
            f"Unsupported model output type: {type(prediction)}"
        )
    
    ##########################################################
    # LOSS
    ##########################################################

    def compute_loss(self, prediction, target, failure=None):
        # 1. Desempacotamento ultra-agressivo baseado em tipo bruto
        forecast_pred = None
        failure_pred = None

        if isinstance(prediction, dict):
            forecast_pred = prediction.get("forecast")
            failure_pred = prediction.get("failure", None)
        elif isinstance(prediction, (tuple, list)):
            # Se for uma tupla/lista, o primeiro elemento é SEMPRE o forecast
            forecast_pred = prediction[0]
            if len(prediction) > 1:
                failure_pred = prediction[1]
        else:
            # Se já for um Tensor direto
            forecast_pred = prediction

        # 2. Varredura recursiva de segurança (caso o forecast ainda venha envelopado em outra tupla)
        while isinstance(forecast_pred, (tuple, list)):
            forecast_pred = forecast_pred[0]
            
        while isinstance(failure_pred, (tuple, list)):
            failure_pred = failure_pred[0]

        # 3. VALIDAÇÃO DE TIPO
        if not isinstance(forecast_pred, torch.Tensor):
            raise TypeError(f"[ERRO CRÍTICO] forecast_pred não é um Tensor! Tipo real: {type(forecast_pred)}.")

        # 4. CORREÇÃO DE SHAPE (Ajuste do horizonte de previsão)
        # Se a rede retornar [batch, seq_len, features] e o target for [batch, features],
        # extraímos apenas o último passo temporal da sequência predita.
        if forecast_pred.dim() == 3 and target.dim() == 2:
            forecast_pred = forecast_pred[:, -1, :]

        # Calcula a perda de forecast usando apenas o tensor puro no formato correto
        forecast_loss = self.criterion(forecast_pred, target)
        
        # 5. Cálculo da perda multi-tarefa puxando o critério do experimento
        failure_criterion = getattr(self.experiment, 'failure_loss', None)
        
        if failure_pred is not None and failure is not None and failure_criterion is not None:
            # Garante que as dimensões do target de falha batam com a saída do modelo
            if isinstance(failure_pred, torch.Tensor):
                if failure_pred.dim() != failure.dim():
                    target_fail = failure.float().unsqueeze(-1) if failure_pred.dim() > failure.dim() else failure.float().squeeze(-1)
                else:
                    target_fail = failure.float()
                    
                f_loss = failure_criterion(failure_pred, target_fail)
                
                f_weight = getattr(self.experiment, 'forecast_weight', 1.0)
                fail_weight = getattr(self.experiment, 'failure_weight', 1.0)
                return (f_weight * forecast_loss) + (fail_weight * f_loss)
            
        return forecast_loss

    ##########################################################
    # TRAIN
    ##########################################################

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for X, y, failure in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            # Garante dimensão correta para classificação binária/multitask
            failure = failure.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Correção do autocast especificando o tipo de dispositivo (device_type)
            with autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.use_amp
            ):
                prediction = self.model(X)
                loss = self.compute_loss(prediction, y, failure)

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    ##########################################################
    # VALIDATION
    ##########################################################

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

    ##########################################################
    # BEST MODEL
    ##########################################################

    def update_best_model(self, val_loss, epoch):
        if val_loss < self.best_loss:
            print(
                f"Validation improved "
                f"{self.best_loss:.6f}"
                f" -> "
                f"{val_loss:.6f}"
            )
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.history["best_epoch"] = epoch + 1
            self.wait = 0
        else:
            self.wait += 1
            print(f"No improvement ({self.wait}/{self.patience})")

        return self.wait >= self.patience

    ##########################################################
    # TRAINING LOOP
    ##########################################################

    def fit(self):
        print()
        print("=" * 60)
        print(self.experiment.name)
        print("=" * 60)
        print()

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
                if hasattr(self.scheduler, "step"):
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