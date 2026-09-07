import torch
import torch.nn as nn
import torch.optim as optim

from src.models.attention_lstnet import AttentionLSTNet
from src.data.dataloader import get_dataloader

from src.training.runner import ExperimentRunner

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, val_loader, test_loader = get_dataloader()

model = AttentionLSTNet(
    input_size=12,
    hidden_size=100,
    output_size=12
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)

runner = ExperimentRunner(

    experiment_name="a_lstnet",

    model=model,

    train_loader=train_loader,

    val_loader=val_loader,

    test_loader=test_loader,

    criterion=criterion,

    optimizer=optimizer,

    device=device

)

runner.run()
