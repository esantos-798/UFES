import torch
import torch.nn as nn
from src.models.transformer_encoder import TransformerEncoder  # Importa o bloco corrigido

class MultiTaskLSTNetTransformer(nn.Module):
    def __init__(self, experiment):
        super().__init__()
        
        # Extração de hiperparâmetros da configuração do experimento
        self.input_size = experiment.input_size     # Número de features de entrada
        self.output_size = experiment.output_size   # Janela de forecasting (ex: 12)
        self.hidden_size = experiment.hidden_size   # Dimensão oculta (d_model)
        
        # Recupera parâmetros específicos do Transformer ou assume fallbacks seguros
        nhead = getattr(experiment, 'nhead', 4)
        num_layers = getattr(experiment, 'num_layers', 2)
        dropout = getattr(experiment, 'dropout', 0.2)
        
        # Recupera parâmetros da Convolução inicial da LSTNet
        cnn_channels = getattr(experiment, 'cnn_channels', 32)
        kernel_size = getattr(experiment, 'kernel_size', 3)
        
        # ----------------------------------------------------------------------
        # COMPONENTE 1: Camada Convolucional (Padrões Locais da LSTNet)
        # ----------------------------------------------------------------------
        # PyTorch Conv1d espera: [Batch, Channels, Seq_Len]
        # Adiciona padding para manter o tamanho da sequência idêntico à entrada
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout)
        
        # ----------------------------------------------------------------------
        # COMPONENTE 2: Transformer Encoder (Memória de Longo Prazo)
        # ----------------------------------------------------------------------
        # A entrada do Transformer receberá a quantidade de canais da CNN anterior
        self.transformer = TransformerEncoder(
            input_size=cnn_channels,
            d_model=self.hidden_size,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # ----------------------------------------------------------------------
        # COMPONENTE 3: Cabeças de Saída Multitarefa (Multitask Heads)
        # ----------------------------------------------------------------------
        # Janela temporal de entrada original obtida via dados (ex: se X tem 12 passos)
        # A saída do transformer_encoder mantém [Batch, Seq_Len, d_model]
        
        # Cabeça 1: Forecasting (Previsão Contínua)
        # Mapeia a dimensão oculta do tempo de volta para o tamanho de saída desejado
        self.forecast_head = nn.Linear(self.hidden_size, self.input_size)
        
        # Cabeça 2: Classificação (Predição de Falhas Binária)
        # Colapsamos o tempo via pooling e aplicamos uma projeção linear de saída única
        self.class_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # Shape inicial de x (DataLoader): [Batch, Seq_Len, Input_Size]
        batch_size, seq_len, _ = x.size()
        
        # 1. Ajusta para Conv1d: [Batch, Input_Size, Seq_Len]
        c_out = x.permute(0, 2, 1)
        c_out = self.conv(c_out)
        c_out = self.relu(c_out)
        c_out = self.conv_dropout(c_out)
        
        # 2. Retorna para o formato sequencial: [Batch, Seq_Len, cnn_channels]
        c_out = c_out.permute(0, 2, 1)
        
        # 3. Processamento via Self-Attention com Positional Encoding
        # Retorna: [Batch, Seq_Len, hidden_size]
        t_out = self.transformer(c_out, return_sequence=True)
        
        # ----------------------------------------------------------------------
        # Divisão das Tarefas
        # ----------------------------------------------------------------------
        
        # Tarefa A: Forecasting
        # Aplicamos a camada linear sobre cada passo temporal da sequência
        forecast_pred = self.forecast_head(t_out) # Saída: [Batch, Seq_Len, Input_Size]
        
        # Tarefa B: Classificação (Predição de Falha)
        # Redução da dimensão temporal através da média global (Pooling)
        pooled_features = torch.mean(t_out, dim=1) # Saída: [Batch, hidden_size]
        class_logits = self.class_head(pooled_features) # Saída: [Batch, 1]
        
        # Retorna ambas as predições de forma limpa para o seu Trainer coletar as Losses
        return forecast_pred, class_logits