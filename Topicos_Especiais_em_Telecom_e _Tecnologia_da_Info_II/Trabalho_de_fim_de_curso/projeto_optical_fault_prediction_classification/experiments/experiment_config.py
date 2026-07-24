from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

@dataclass
class ExperimentConfig:
    # Identificação
    model: str
    dataset: str

    # Multitask Weights
    forecast_weight: float
    failure_weight: float
    pos_weight: float
    alpha: float

    # Treino e Otimização
    batch_size: int = 64
    epochs: int = 50
    patience: int = 7
    lr: float = 0.001
    task: str = "forecast"
    
    # Hiperparâmetros da Arquitetura do Modelo
    input_size: int = 12
    hidden_size: int = 64
    output_size: int = 12
    num_layers: int = 2
    dropout: float = 0.2
    d_model: int = 64
    nhead: int = 4
    cnn_channels: int = 32
    kernel_size: int = 3

    # Reprodução
    random_seed: int = 42

    # Saída Dinâmica
    output_dir: str = ""

    # Critérios de Perda
    criterion: nn.Module = field(default_factory=nn.MSELoss)
    failure_loss: nn.Module = field(init=False)

    def __post_init__(self):
        # 1. Cria dinamicamente a função de perda de classificação ponderada
        self.failure_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32)
        )
        
        # 2. Define um nome descritivo automático para o experimento
        self.name = f"{self.model}_{self.dataset}_f{self.failure_weight}_p{self.pos_weight}_a{self.alpha}"
        
        # 3. Define a estrutura de diretórios de saída se não for provida manualmente
        if not self.output_dir:
            self.output_dir = str(Path("results") / "runs" / self.name)