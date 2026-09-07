import torch
import torch.nn as nn


class MultiTaskLSTNet(nn.Module):

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

        # Dentro do __init__ de MultiTaskLSTNet:
        # Descobre dinamicamente o tamanho do output do kernel convolucional
        conv_out_features = config.input_size - config.kernel_size + 1 
        # Se o kernel bater exatamente com o input_size (como kernel=12 e input=12), vira 1
        conv_out_features = max(1, conv_out_features) 

        # Multiplica o número de canais da CNN pelo que sobrou das features
        gru_input_dim = config.cnn_channels * conv_out_features 

        # Agora instancie o GRU usando essa variável dinâmica
        self.gru = nn.GRU(gru_input_dim, config.hidden_size, batch_first=True)
        self.skip = nn.GRU(gru_input_dim, config.hidden_size, batch_first=True)

        ####################################################
        # GRU principal
        ####################################################

        #self.gru = nn.GRU(
        #    cnn_channels,
        #    hidden_size,
        #    batch_first=True
        #)
        #self.gru = nn.GRU(input_size=96, hidden_size=64, batch_first=True)
        
        ####################################################
        # Skip GRU
        ####################################################

        #self.skip = nn.GRU(
        #    cnn_channels,
        #    hidden_size,
        #    batch_first=True
        #)
        #self.skip = nn.GRU(input_size=96, hidden_size=64, batch_first=True) # Se houver a skip
        ####################################################
        # Camada compartilhada
        ####################################################

        fusion_size = hidden_size * 2

        ####################################################
        # Forecast Head
        ####################################################

        #self.forecast_head = nn.Sequential

        #    nn.Linear(
        #        fusion_size,
        #        hidden_size
        #    ),

        #    nn.ReLU(),

        #    nn.Dropout(dropout),

        #    nn.Linear(
        #        hidden_size,
        #        output_size
        #    )
        #)
        self.forecast_head = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=output_size) # Garanta que está 'output_size' aqui, e não 1!
        )
        ####################################################
        # Failure Head
        ####################################################

        self.failure_head = nn.Sequential(

            nn.Linear(
                fusion_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                1
            )

            # NÃO colocar Sigmoid aqui
            # BCEWithLogitsLoss já aplica internamente
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
    
        # Permuta para colocar a dimensão temporal (seq_out) na frente: [64, seq_out, 32, 3]
        c = c.permute(0, 2, 1, 3).contiguous()
    
        # Achata as dimensões de canais e features residuais em uma só: [64, seq_out, 32 * 3]
        c = c.view(batch_size, seq_out, channels * feat_out) 
        # ----------------------------

        # 3. Agora o tensor está em 3D e pronto para as GRUs
        _, h = self.gru(c)
    
        # Se você também utiliza a camada 'self.skip', ela deve receber o mesmo tensor 'c'
        # _, h_skip = self.skip(c) 
    
        # ... restante do seu código (as heads de forecast e failure)
        h = h[-1]

        ####################################################
        # Skip GRU
        ####################################################

        _, hs = self.skip(c)

        hs = hs[-1]

        ####################################################
        # Fusão
        ####################################################

        features = torch.cat(
            [h, hs],
            dim=1
        )

        ####################################################
        # Heads
        ####################################################

        forecast = self.forecast_head(features)

        failure = self.failure_head(features)

        return {
            "forecast": forecast,
            "failure": failure
        }