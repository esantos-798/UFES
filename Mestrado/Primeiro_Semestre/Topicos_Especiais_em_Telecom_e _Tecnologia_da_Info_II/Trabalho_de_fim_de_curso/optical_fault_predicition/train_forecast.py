import torch
import torch.nn as nn
import torch.optim as optim

from src.models.lstm import LSTM
from src.data.dataloader import get_dataloader

from src.training.forecast_trainer import ForecastTrainer

from src.training.forecast_evaluator import ForecastEvaluator


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Data
# =========================

train_loader, val_loader, test_loader = get_dataloader(
    task="forecast"
)


# =========================
# Model
# =========================

model = LSTM(

    input_size=12,

    hidden_size=100,

    num_layers=1,

    output_size=12

).to(device)


# =========================
# Loss
# =========================

criterion = nn.HuberLoss()


# =========================
# Optimizer
# =========================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
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

    device=device,

    patience=5,

    checkpoint_path="best_lstm_forecast.pt"

)


EPOCHS = 30


for epoch in range(EPOCHS):

    train_loss = trainer.train_epoch()

    val_loss = trainer.validate()


    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} "
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


# carregar melhor modelo

model.load_state_dict(
    torch.load(
        "best_lstm_forecast.pt",
        weights_only=True
    )
)


evaluator = ForecastEvaluator(
    model=model,
    test_loader=test_loader,
    device=device
)


results = evaluator.evaluate()


print("\n===== FORECAST RESULTS =====")

print(
    f"MSE     : {results['MSE']:.6f}"
)

print(
    f"RMSE    : {results['RMSE']:.6f}"
)

print(
    f"MAE     : {results['MAE']:.6f}"
)

print(
    f"R²      : {results['R2']:.6f}"
)

print(
    f"Pearson : {results['Pearson']:.6f}"
    )