import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from xgboost import XGBRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import random
import os

# Configuração de reprodutibilidade e hardware
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
seed_everything()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. CARREGAMENTO E DIVISÃO DO DATASET REAL
# ==========================================
def preparar_dados_reais(csv_path, window_size=10, batch_size=64):
    df = pd.read_csv(csv_path)
    
    # 1. Remover colunas de identificação/tempo
    colunas_remover = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'index' in c.lower()]
    df_limpo = df.drop(columns=colunas_remover, errors='ignore')
    df_numerico = df_limpo.select_dtypes(include=[np.number])
    
    # Tratar valores inconsistentes
    df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan)
    df_numerico = df_numerico.ffill().bfill().fillna(0.0)
    
    # ... dentro de preparar_dados_reais, após ffill().bfill() ...
    df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    
    # --- CORREÇÃO DA TRANSFORMAÇÃO LOGARÍTMICA ---
    # Aplica o -log10 APENAS se a coluna explicitamente contiver "ber" ou "fec" no nome
    for col in df_numerico.columns:
        if 'ber' in col.lower() or 'fec' in col.lower():
            # Remove zeros absolutos para evitar log10(0) = -inf
            df_numerico[col] = df_numerico[col].replace(0, 1e-12)
            df_numerico[col] = -np.log10(df_numerico[col])
            print(f"-> [SUCESSO] Aplicada transformação -log10 na coluna correta: '{col}'")
            
    # Trata NaNs/Infs residuais se houverem
    df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    features = df_numerico.values

    n_samples = len(features)
    
    # 2. Divisão exata do Artigo (70% Treino / 10% Validação / 20% Teste)
    split_train = int(n_samples * 0.70)
    split_val = int(n_samples * 0.80)
    
    train_raw = features[:split_train]
    val_raw = features[split_train:split_val]
    test_raw = features[split_val:]
    
    # 3. CRIAÇÃO DO GABARITO DE FALHAS (Injetado apenas no Teste conforme o Artigo)
    # Ciclo do Artigo: 4 minutos normais (~69 amostras), 1 minuto falha (~17 amostras)
    test_len = len(test_raw)
    test_labels = np.zeros(test_len)
    
    amostras_normais = 69
    amostras_falha = 17
    periodo_ciclo = amostras_normais + amostras_falha
    
    for idx in range(test_len):
        pos_no_ciclo = idx % periodo_ciclo
        if pos_no_ciclo >= amostras_normais:
            test_labels[idx] = 1.0  # Marca como Hard Failure induzida
            
    print(f"-> Total de features de telemetria mantidas para o LSTNet: {features.shape[1]}")
    print(f"-> Amostras de falha geradas ciclicamente no teste (Artigo): {int(np.sum(test_labels))} de {test_len}")

    # 4. Normalização Robusta
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)
    
    train_scaled = np.nan_to_num(train_scaled, nan=0.0)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0)
    
    def create_sliding_windows(data, window):
        X, Y = [], []
        for i in range(len(data) - window):
            X.append(data[i : i + window])
            Y.append(data[i + window])
        return np.array(X), np.array(Y)
        
    X_tr, Y_tr = create_sliding_windows(train_scaled, window_size)
    X_va, Y_va = create_sliding_windows(val_scaled, window_size)
    X_te, Y_te = create_sliding_windows(test_scaled, window_size)
    
    # Ajustar o vetor de labels para casar com o deslocamento da janela deslizante do teste
    test_labels_ajustado = test_labels[window_size:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(Y_va, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(Y_te, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_labels_ajustado, features.shape[1]

# ==========================================
# 2. IMPLEMENTAÇÃO DO LSTNET MODIFICADO DO ARTIGO
# ==========================================
class LSTNetModificado(nn.Module):
    def __init__(self, window, hidRNN, hidCNN, CNN_kernel, skip, hidSkip, num_features):
        super().__init__()
        self.window = window
        self.skip = skip
        self.num_features = num_features
        
        # 1. Componente Convolucional (Filtro espacial-temporal)
        self.conv1 = nn.Conv2d(1, hidCNN, kernel_size=(CNN_kernel, num_features))
        
        # 2. Recorrência Padrão (GRU)
        self.GRU1 = nn.GRU(hidCNN, hidRNN, batch_first=True)
        
        # 3. Recorrência Skip (Padrões periódicos)
        self.hidSkip_dim = (window - CNN_kernel + 1) // skip
        if self.hidSkip_dim > 0:
            self.GRU_skip = nn.GRU(hidCNN, hidSkip, batch_first=True)
            self.linear_skip = nn.Linear(hidRNN + skip * hidSkip, num_features)
        else:
            self.linear_skip = nn.Linear(hidRNN, num_features)
            
        # 4. Componente Autorregressiva Linear Global (Filtro de Escala do Artigo)
        self.ar = nn.Linear(window, 1)
        self.output_layer = nn.Linear(num_features, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN
        c = x.view(-1, 1, self.window, self.num_features)
        c = F.relu(self.conv1(c))
        c = torch.squeeze(c, 3).permute(0, 2, 1)
        
        # GRU Padrão
        _, r = self.GRU1(c)
        r = torch.squeeze(r, 0)
        
        # Skip-GRU
        if self.hidSkip_dim > 0:
            s_input = c[:, -self.hidSkip_dim * self.skip :, :].contiguous()
            s_input = s_input.view(batch_size, self.hidSkip_dim, self.skip, -1)
            s_input = s_input.permute(0, 2, 1, 3).contiguous().view(-1, self.hidSkip_dim, c.size(-1))
            
            _, s = self.GRU_skip(s_input)
            s = torch.squeeze(s, 0)
            s = s.view(self.skip, batch_size, -1).permute(1, 0, 2).contiguous().view(batch_size, -1)
            res = torch.cat((r, s), dim=1)
        else:
            res = r
            
        neural_out = self.linear_skip(res)
        
        # AR Linear Global (Exatamente como formulado originalmente)
        x_ar = x.permute(0, 2, 1).contiguous().view(-1, self.window)
        x_ar = self.ar(x_ar).view(batch_size, self.num_features)
        
        # Fusão com ReLU Clássica
        h_out = neural_out + x_ar
        return F.relu(self.output_layer(h_out))


def train_torch_model_l1(model, train_loader, epochs=30):
    # Otimizador Adam padrão com L1Loss puro (idêntico ao paper para prever o comportamento normal estável)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss() 
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Ignora lotes residuais se o dataloader trouxer tamanhos inconsistentes no fim do loop
            if x.size(0) != train_loader.batch_size:
                continue
                
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

# ==========================================
# 3. ROTINAS DE VALIDAÇÃO E EXECUÇÃO
# ==========================================
def train_torch_model_l1(model, train_loader, epochs=30):
    # Adicionamos weight_decay para evitar que os pesos explodam ou causem instabilidade
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    #criterion = nn.L1Loss() 
    criterion = nn.MSELoss() # <--- Mudado de L1Loss para MSELoss
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

def evaluate_regression_validation(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            
    # Concatena todos os batches em tensores únicos
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Cálculos robustos com Scikit-Learn utilizando os arrays totalmente alinhados
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Pearson para matrizes multivariadas (média das correlações por coluna)
    pearsons = []
    for i in range(all_targets.shape[1]):
        if np.std(all_targets[:, i]) > 1e-6 and np.std(all_preds[:, i]) > 1e-6:
            r, _ = pearsonr(all_targets[:, i], all_preds[:, i])
            pearsons.append(r)
    pearson_avg = np.mean(pearsons) if pearsons else 0.0
    
    # Limiar dinâmico idêntico ao do Artigo (Percentil 98.5 do erro absoluto médio)
    erro_abs_individual = np.mean(np.abs(all_targets - all_preds), axis=1)
    threshold = np.percentile(erro_abs_individual, 98.5)
    
    return mse, r2, pearson_avg, threshold


def evaluate_and_get_metrics(model, test_loader, true_labels, lookback, threshold):
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            # Erro quadrático médio por amostra (distância euclidiana multivariada)
            error = torch.sqrt(torch.mean((preds - y) ** 2, dim=1))
            all_errors.extend(error.cpu().numpy())
            
    preds_binary = np.array([1 if e > threshold else 0 for e in all_errors])
    
    # Sincroniza o tamanho exato usando as últimas posições temporais do gabarito
    n_preds = len(preds_binary)
    aligned_labels = true_labels[-n_preds:]
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        aligned_labels, preds_binary, average='binary', zero_division=0
    )
    return f1, prec, rec

if __name__ == "__main__":
    print("--- Inicializando Pipeline REAL: Dataset Hard do Artigo ---")
    import time  # Certifique-se de importar o time no topo ou aqui
    # Altere aqui para o nome exato do seu arquivo CSV real
    csv_nome = "SoftFailure_dataset.csv" 
    
    if not os.path.exists(csv_nome):
        print(f"Erro: O arquivo '{csv_nome}' não foi encontrado na pasta. Coloque-o na mesma raiz do script.")
    else:
        tempo_inicio = time.time()
        lookback = 24
        train_loader, val_loader, test_loader, test_labels, num_features = preparar_dados_reais(csv_path=csv_nome, window_size=lookback)
        
        print(f"Dataset carregado com sucesso. Número de features dinâmicas detectadas: {num_features}")
        print("Treinando o modelo LSTNet Modificado com as diretrizes do Artigo...")
        
        # Instanciação seguindo os hiperparâmetros sintonizados
        # Instanciação aumentando o skip para estabilizar a componente autoregressiva
        model_lstnet = LSTNetModificado(
            window=lookback, 
            hidRNN=32, 
            hidCNN=32, 
            CNN_kernel=3, 
            skip=4,        # Ajustado para casar com lookback de 24
            hidSkip=16, 
            num_features=num_features
        ).to(device)
        
        print("Treinando por 30 épocas com sintonização fina...")
        train_torch_model_l1(model_lstnet, train_loader, epochs=30)
        
        train_torch_model_l1(model_lstnet, train_loader, epochs=10) # 10 épocas para convergência em dados reais
        
        mse, r2, pearson, thresh = evaluate_regression_validation(model_lstnet, val_loader)
        f1, p, r = evaluate_and_get_metrics(model_lstnet, test_loader, test_labels, lookback, thresh)
        
        tempo_total = time.time() - tempo_inicio
        minutos = int(tempo_total // 60)
        segundos = int(tempo_total % 60)

        print("\n" + "="*54)
        print(" RESULTADOS DO LSTNET MODIFICADO (DADOS REAIS)")
        print("="*54)
        print(f"MSE de Validação: {mse:.5f}")
        print(f"R2-Score de Validação: {r2:.4f}")
        print(f"Pearson (r) de Validação: {pearson:.4f}")
        print("-"*54)
        print(f"F1-Score no Teste: {f1:.4f}")
        print(f"Precision no Teste: {p:.4f}")
        print(f"Recall no Teste: {r:.4f}")
        print("="*54)
        print(f"Tempo de Execução Total: {minutos} min {segundos} seg")  # <--- NOVA LINHA
        print("="*54)