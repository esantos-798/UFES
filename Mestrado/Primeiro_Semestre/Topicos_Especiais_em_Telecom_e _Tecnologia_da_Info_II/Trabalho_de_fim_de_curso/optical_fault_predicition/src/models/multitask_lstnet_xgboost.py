import torch
import torch.nn as nn
import numpy as np
from xgboost import XGBClassifier

class MultiTaskLSTNetXGBoost(nn.Module):
    def __init__(self, window, num_features, hidCNN, hidRNN, hidSkip, skip, highway_window, num_classes=1):
        super(MultiTaskLSTNetXGBoost, self).__init__()
        self.window = window
        self.num_features = num_features
        
        # 1. Componente Convolucional (Espacial / Correlações locais entre variáveis)
        self.conv1 = nn.Conv2d(1, hidCNN, kernel_size=(3, num_features))
        
        # 2. Componente Recorrente (Temporal de Curto Prazo)
        self.rnn1 = nn.GRU(hidCNN, hidRNN, batch_first=True)
        
        # 3. Recorrência com Salto (Temporal de Longo Prazo / Sazonalidade)
        self.skip = skip
        self.rnn2 = nn.GRU(hidCNN, hidSkip, batch_first=True)
        
        # 4. Camada Extratora de Features Latentes (Onde o XGBoost vai beber)
        # O tamanho combina a saída da rnn1 + rnn2 (multiplicada pelo número de saltos possíveis)
        self.num_skip_clones = int((window - 3 + 1) / skip)
        self.feature_dim = hidRNN + (self.num_skip_clones * hidSkip)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 5. Cabeças de Saída Tradicionais (MultiTask Padrão)
        self.forecast_head = nn.Linear(64, num_features)
        self.classify_head = nn.Linear(64, num_classes)
        
        # 6. Componente Autoregressivo Linear (Highway)
        self.highway = nn.Linear(highway_window, 1)
        
        # Atributo para armazenar o classificador XGBoost treinado externamente
        self.xgb_model = None

    def forward(self, x):
        # x shape: [batch, window, num_features]
        batch_size = x.size(0)
        
        # Alinha para a CNN: [batch, channel=1, height=window, width=num_features]
        c = x.view(-1, 1, self.window, self.num_features)
        c = torch.relu(self.conv1(c))
        c = c.squeeze(3) # [batch, hidCNN, new_window]
        c = c.permute(0, 2, 1) # [batch, new_window, hidCNN]
        
        # RNN curta
        _, r = self.rnn1(c)
        r = r.squeeze(0) # [batch, hidRNN]
        
        # RNN com Salto (Skip)
        new_window = c.size(1)
        if self.skip > 0 and new_window >= self.skip:
            # Reorganiza o tensor para agrupar elementos espaçados pelo valor de 'skip'
            s = c[:, :int(new_window/self.skip)*self.skip, :]
            s = s.view(batch_size, -1, self.skip, c.size(2))
            s = s.permute(0, 2, 1, 3)
            s = s.contiguous().view(-1, s.size(2), s.size(3))
            _, s = self.rnn2(s)
            s = s.squeeze(0)
            s = s.view(batch_size, -1) # [batch, num_skip_clones * hidSkip]
            out_latente = torch.cat((r, s), 1)
        else:
            out_latente = r
            
        # Extrai vetor denso de representação temporal
        features = self.feature_extractor(out_latente)
        
        # Task 1: Previsão de Séries Temporais + Componente AutoRegressivo
        forecast = self.forecast_head(features)
        # Ajusta componente linear simples sobre o final da janela
        ans = x[:, -1, :] # Fallback simplificado para o formato do Highway
        forecast = forecast + ans 
        
        # Task 2: Classificação via Rede Neural (Deep Target)
        logits = self.classify_head(features)
        probs = torch.sigmoid(logits)
        
        return forecast, probs, features

    def fit_xgb(self, train_features, train_labels):
        """Treina o modelo XGBoost usando as features profundas extraídas da LSTNet"""
        print("-> Treinando cabeça híbrida XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=5.0, # Ajuste para o desbalanceamento óptico
            eval_metric="logloss",
            random_state=42
        )
        self.xgb_model.fit(train_features, train_labels)

    def predict_xgb(self, features):
        """Predição usando o modelo XGBoost acoplado"""
        if self.xgb_model is None:
            raise ValueError("O modelo XGBoost precisa ser treinado primeiro com fit_xgb().")
        return self.xgb_model.predict_proba(features)[:, 1]