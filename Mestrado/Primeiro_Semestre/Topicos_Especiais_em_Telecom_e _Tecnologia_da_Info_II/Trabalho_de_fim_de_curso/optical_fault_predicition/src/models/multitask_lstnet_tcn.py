import torch
import torch.nn as nn
from src.models.tcn import TCN  # Importa o bloco TCN corrigido

class MultiTaskLSTNetTCN(nn.Module):
    def __init__(self, experiment):
        super().__init__()
        
        # Extração de hiperparâmetros da configuração do experimento
        self.input_size = experiment.input_size     # Número de features de entrada
        self.output_size = experiment.output_size   # Janela de forecasting (ex: 12)
        self.hidden_size = experiment.hidden_size   # Dimensão oculta dos blocos temporais
        
        # Recupera parâmetros específicos da TCN ou assume fallbacks seguros
        num_levels = getattr(experiment, 'num_levels', 3)
        dropout = getattr(experiment, 'dropout', 0.2)
        
        # Recupera parâmetros da Convolução inicial da LSTNet
        cnn_channels = getattr(experiment, 'cnn_channels', 32)
        kernel_size = getattr(experiment, 'kernel_size', 3)
        
        # ----------------------------------------------------------------------
        # COMPONENTE 1: Camada Convolucional Inicial (Padrões Locais da LSTNet)
        # ----------------------------------------------------------------------
        # PyTorch Conv1d espera: [Batch, Channels, Seq_Len]
        # Aplica padding para manter o tamanho da sequência perfeitamente alinhado
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
        # COMPONENTE 2: TCN (Histórico de Longo Prazo Causal)
        # ----------------------------------------------------------------------
        # A TCN receberá na entrada a quantidade de canais gerados pela CNN
        # Deixamos output_size=None para usá-la puramente como extratora de features
        self.tcn = TCN(
            input_size=cnn_channels,
            hidden_size=self.hidden_size,
            output_size=None,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # ----------------------------------------------------------------------
        # COMPONENTE 3: Cabeças de Saída Multitarefa (Multitask Heads)
        # ----------------------------------------------------------------------
        # A TCN corrigida com return_sequence=True devolve [Batch, Seq_Len, hidden_size]
        
        # Cabeça 1: Forecasting (Previsão Contínua)
        self.forecast_head = nn.Linear(self.hidden_size, self.input_size)
        
        # Cabeça 2: Classificação (Predição de Falhas Binária)
        # Para a TCN, em vez da média, extrair o último passo temporal é o ideal
        # porque as convoluções causais acumulam todo o passado no último frame
        self.class_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # Shape inicial de x (DataLoader): [Batch, Seq_Len, Input_Size][cite: 9]
        batch_size, seq_len, _ = x.size()
        
        # 1. Ajusta para Conv1d: [Batch, Input_Size, Seq_Len]
        c_out = x.permute(0, 2, 1)
        c_out = self.conv(c_out)
        c_out = self.relu(c_out)
        c_out = self.conv_dropout(c_out)
        
        # 2. Prepara para a TCN esperando: [Batch, Seq_Len, cnn_channels][cite: 9]
        c_out = c_out.permute(0, 2, 1)
        
        # 3. Processamento via Convoluções Causais Dilatadas
        # Retorna: [Batch, Seq_Len, hidden_size]
        tcn_out = self.tcn(c_out, return_sequence=True)
        
        # ----------------------------------------------------------------------
        # Divisão das Tarefas
        # ----------------------------------------------------------------------
        
        # Tarefa A: Forecasting
        # Mapeia passo a passo os estados temporais da sequência
        forecast_pred = self.forecast_head(tcn_out)  # Saída: [Batch, Seq_Len, Input_Size]
        
        # Tarefa B: Classificação (Predição de Falha)
        # Extrai o último frame temporal da TCN (resumo causal perfeito do passado)[cite: 9]
        last_step_features = tcn_out[:, -1, :]        # Saída: [Batch, hidden_size][cite: 9]
        class_logits = self.class_head(last_step_features)  # Saída: [Batch, 1]
        
        return forecast_pred, class_logits