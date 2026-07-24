import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x = [batch, seq, hidden]

        scores = self.attn(x)

        weights = F.softmax(scores, dim=1)

        context = torch.sum(weights * x, dim=1)

        return context, weights
    
class MultiTaskLSTNetAttention(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        cnn_channels=32,
        kernel_size=3,
        hidden_size=100,
        dropout=0.2
    ):

        super().__init__()

        self.output_size = output_size

        ####################################################
        # CNN temporal
        ####################################################

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(kernel_size, input_size)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # CÁLCULO DINÂMICO DAS DIMENSÕES (CORREÇÃO DO BUG)
        ####################################################
        # Descobre dinamicamente o tamanho do output do kernel convolucional nas features
        # Como a largura do kernel é igual a input_size, a dimensão de features vira sempre 1
        conv_out_features = input_size - input_size + 1 
        conv_out_features = max(1, conv_out_features) 

        # Multiplica o número de canais da CNN pelo que sobrou das features
        gru_input_dim = cnn_channels * conv_out_features 

        ####################################################
        # GRU principal e Skip GRU (Dinâmicas)
        ####################################################

        self.gru = nn.GRU(gru_input_dim, hidden_size, batch_first=True)
        self.skip = nn.GRU(gru_input_dim, hidden_size, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        ####################################################
        # Camada compartilhada e Heads
        ####################################################

        fusion_size = hidden_size * 2

        # Forecast Head
        self.forecast_head = nn.Sequential(
            nn.Linear(in_features=fusion_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        
        # Failure Head
        self.failure_head = nn.Sequential(
            nn.Linear(in_features=fusion_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=1)
            # Sem Sigmoid aqui, o BCEWithLogitsLoss já cuida disso
        )


    def forward(self, x):
        # x formato original: [batch_size, seq_len, num_features] -> [64, 30, 12]
    
        # 1. Adiciona a dimensão de canal (1) exigida pela Conv2d
        c = x.unsqueeze(1) # Resultado: [64, 1, 30, 12]
    
        # 2. Passa pelas camadas convolucionais
        c = self.conv(c)
        c = self.relu(c)
        c = self.dropout(c) # Resultado: [64, 32, seq_out, feature_out]
    
        # --- CORREÇÃO DO ERRO 4D ---
        # Captura as dimensões dinamicamente
        batch_size, channels, seq_out, feat_out = c.size()
    
        # Permuta para colocar a dimensão temporal (seq_out) na frente
        c = c.permute(0, 2, 1, 3).contiguous()
    
        # Achata as dimensões de canais e features residuais em uma só
        c = c.view(batch_size, seq_out, channels * feat_out) 
        # ----------------------------

        # 3. Agora o tensor está em 3D e pronto para as GRUs
        gru_out, _ = self.gru(c)

        h, attention = self.attention(gru_out)

        # 4. Passa pela Skip GRU
        skip_step = 2

        skip_input = c[:, ::skip_step, :]

        _, hs = self.skip(skip_input)

        hs = hs[-1]
        #_, hs = self.skip(c)
        #hs = hs[-1]

        ####################################################
        # Fusão e Predição
        ####################################################

        features = torch.cat([h, hs], dim=1)

        forecast = self.forecast_head(features)
        failure = self.failure_head(features)

        return {

            "forecast": forecast,

            "failure": failure,

            "attention": attention
        }