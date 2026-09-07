import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, r2_score
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
# 1. CLASSE DE DATASET E ESTRUTURAÇÃO
# ==========================================
class OpticalDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return x, y

def get_loaders(data, lookback, batch_size=64):
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    train_loader = DataLoader(OpticalDataset(train_data, lookback), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OpticalDataset(val_data, lookback), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(OpticalDataset(test_data, lookback), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==========================================
# 2. GERADOR DE DATASET SINTÉTICO (HARD)
# ==========================================
def generate_synthetic_hard_dataset():
    n_samples = 10945
    n_features = 12
    t = np.linspace(0, 500, n_samples)
    
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        signal = 0.4 * np.sin(t * 0.1 + i) + 0.2 * np.cos(t * 0.05 - i)
        noise = 0.12 * np.random.normal(size=n_samples)
        data[:, i] = 0.5 + signal + noise
        
    test_start_idx = int(n_samples * 0.80)
    labels = np.zeros(n_samples)
    
    step = 0
    is_failure_cycle = False
    for idx in range(test_start_idx, n_samples):
        if step % 120 == 0 and step > 0:
            is_failure_cycle = True
        if is_failure_cycle and (step % 150 == 0):
            is_failure_cycle = False
        
        if is_failure_cycle:
            data[idx, 0] -= 0.45  # Queda no Input do Ampli1
            data[idx, 8] -= 0.50  # Queda no OSNR do Card1
            labels[idx] = 1
        step += 1
        
    return data, labels

# ==========================================
# 3. IMPLEMENTAÇÃO DOS 8 MODELOS
# ==========================================
class BaselineRNNs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='LSTM', num_layers=2):
        super().__init__()
        self.model_type = model_type
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'BiLSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            
        mult = 2 if model_type == 'BiLSTM' else 1
        self.fc = nn.Linear(hidden_dim * mult, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTNet(nn.Module):
    def __init__(self, window, hidRNN, hidCNN, CNN_kernel, skip, hidSkip, num_features):
        super().__init__()
        self.window = window
        self.skip = skip
        self.num_features = num_features
        self.conv1 = nn.Conv2d(1, hidCNN, kernel_size=(CNN_kernel, num_features))
        self.GRU1 = nn.GRU(hidCNN, hidRNN, batch_first=True)
        
        self.hidSkip_dim = (window - CNN_kernel + 1) // skip
        if self.hidSkip_dim > 0:
            self.GRU_skip = nn.GRU(hidCNN, hidSkip, batch_first=True)
            self.linear_skip = nn.Linear(hidRNN + skip * hidSkip, num_features)
        else:
            self.linear_skip = nn.Linear(hidRNN, num_features)
            
        self.ar = nn.Linear(window, 1)
        self.output_layer = nn.Linear(num_features, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        c = x.view(-1, 1, self.window, self.num_features)
        c = F.relu(self.conv1(c))
        c = torch.squeeze(c, 3).permute(0, 2, 1)
        
        _, r = self.GRU1(c)
        r = torch.squeeze(r, 0)
        
        if self.hidSkip_dim > 0:
            s_input = c[:, -self.hidSkip_dim * self.skip :, :].contiguous()
            s_input = s_input.view(batch_size, self.hidSkip_dim, self.skip, -1)
            s_input = s_input.permute(0, 2, 1, 3).contiguous().view(-1, self.hidSkip_dim, c.size(-1))
            _, s = self.GRU_skip(s_input)
            s = s.view(self.skip, batch_size, -1).permute(1, 0, 2).contiguous().view(batch_size, -1)
            res = torch.cat((r, s), dim=1)
        else:
            res = r
            
        # ... (mantenha o restante do seu código inicial igual)
        neural_out = self.linear_skip(res)
        x_ar = x.permute(0, 2, 1).contiguous().view(-1, self.window)
        x_ar = self.ar(x_ar).view(-1, self.num_features)
    
        # --- ALTERAÇÃO DO ARTIGO ---
        # Em vez de passar o 'out' direto pela camada final de forma isolada,
        # o artigo soma as duas componentes e passa o resultado pela Fully Connected + ReLU
        h_out = neural_out + x_ar  # Eq. (7) do artigo: h_out = h_D + h_AR
    
        # Retorna a saída final projetada com ativação ReLU
        return F.relu(self.output_layer(h_out))  # Eq. (8) do artigo

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, lookback, embedding_dim=32, n_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim * lookback, input_dim)

    def forward(self, x):
        x_emb = self.input_proj(x)
        out = self.transformer(x_emb)
        out = out.contiguous().view(out.size(0), -1)
        return self.fc(out)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class GCN_LSTM(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim, lookback, adj_matrix):
        super().__init__()
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        self.gcn = GCNLayer(node_features, hidden_dim)
        self.lstm = nn.LSTM(num_nodes * hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_nodes * node_features)
        self.num_nodes = num_nodes
        self.node_features = node_features

    def forward(self, x):
        batch_size, lookback, _ = x.shape
        x_graph = x.view(batch_size, lookback, self.num_nodes, self.node_features)
        
        gcn_out = F.relu(self.gcn(x_graph, self.adj))
        gcn_out = gcn_out.view(batch_size, lookback, -1)
        
        lstm_out, _ = self.lstm(gcn_out)
        pred = self.fc(lstm_out[:, -1, :])
        return pred

# ==========================================
# 4. ROTINAS DE TREINO E ENGINES DE MÉTRICAS
# ==========================================
def train_torch_model(model, train_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.L1Loss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

def evaluate_regression_validation(model, val_loader, is_xgboost=False, xgboost_model=None):
    """
    Calcula MSE, R2 e Pearson no conjunto de VALIDAÇÃO (dados normais)
    """
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            if is_xgboost:
                lstm_p = model(x.to(device)).cpu().numpy()
                xgb_resid = xgboost_model.predict(lstm_p)
                pred = lstm_p + xgb_resid
            else:
                pred = model(x.to(device)).cpu().numpy()
            
            preds.extend(pred)
            targets.extend(y.numpy())
            
    preds, targets = np.array(preds), np.array(targets)
    
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    
    # Pearson médio entre as features preditas e reais
    pearsons = []
    for i in range(targets.shape[1]):
        if np.std(targets[:, i]) > 0 and np.std(preds[:, i]) > 0:
            r, _ = pearsonr(targets[:, i], preds[:, i])
            pearsons.append(r)
    mean_pearson = np.mean(pearsons) if pearsons else 0.0
    
    # Calcular os erros para o cálculo posterior do Threshold
    errors = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
    threshold = np.percentile(errors, 98.5)
    
    return mse, r2, mean_pearson, threshold

def evaluate_and_get_metrics_v2(model, test_loader, true_labels, lookback, threshold, is_xgboost=False, xgboost_model=None):
    errors = []
    
    if is_xgboost:
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                lstm_pred = model(x).cpu().numpy()
                xgb_resid = xgboost_model.predict(lstm_pred)
                final_pred = lstm_pred + xgb_resid
                error = np.sqrt(np.sum((final_pred - y.numpy()) ** 2, axis=1))
                errors.extend(error)
    else:
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                error = torch.sqrt(torch.sum((pred - y) ** 2, dim=1))
                errors.extend(error.cpu().numpy())
                
    preds_binary = [1 if e > threshold else 0 for e in errors]
    aligned_labels = true_labels[lookback:]
    
    prec, rec, f1, _ = precision_recall_fscore_support(aligned_labels, preds_binary, average='binary', zero_division=0)
    return f1, prec, rec

# ==========================================
# 5. LOOP DE EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("--- Inicializando Pipeline de Teste: Dataset Hard ---")
    
    lookback = 10
    num_features = 12
    epochs = 5
    
    raw_data, labels = generate_synthetic_hard_dataset()
    train_loader, val_loader, test_loader = get_loaders(raw_data, lookback)
    
    adj_matrix = np.eye(6)
    adj_matrix[0, 1] = 1
    adj_matrix[1, 2] = 1
    adj_matrix[2, 3] = 1
    adj_matrix[3, 4] = 1
    adj_matrix[4, 5] = 1

    resultados_val = {}
    resultados_test = {}

    # --- Modelos 1 a 4 (Recorrentes Clássicos) ---
    for m_type in ['RNN', 'LSTM', 'BiLSTM', 'GRU']:
        print(f"Treinando e Avaliando: {m_type}...")
        model = BaselineRNNs(num_features, 32, num_features, model_type=m_type).to(device)
        train_torch_model(model, train_loader, epochs)
        
        mse, r2, pearson, thresh = evaluate_regression_validation(model, val_loader)
        resultados_val[m_type] = {"MSE": mse, "R2": r2, "Pearson": pearson}
        
        f1, p, r = evaluate_and_get_metrics_v2(model, test_loader, labels[int(len(labels)*0.80):], lookback, thresh)
        resultados_test[m_type] = {"F1-Score": f1, "Precision": p, "Recall": r}

    # --- Modelo 5: LSTNet ---
    print("Treinando e Avaliando: LSTNet...")
    model_lstnet = LSTNet(window=lookback, hidRNN=32, hidCNN=32, CNN_kernel=3, skip=2, hidSkip=16, num_features=num_features).to(device)
    train_torch_model(model_lstnet, train_loader, epochs)
    mse, r2, pearson, thresh_lstnet = evaluate_regression_validation(model_lstnet, val_loader)
    resultados_val["LSTNet"] = {"MSE": mse, "R2": r2, "Pearson": pearson}
    f1, p, r = evaluate_and_get_metrics_v2(model_lstnet, test_loader, labels[int(len(labels)*0.80):], lookback, thresh_lstnet)
    resultados_test["LSTNet"] = {"F1-Score": f1, "Precision": p, "Recall": r}

    # --- Modelo 6: Transformer ---
    print("Treinando e Avaliando: Transformer...")
    model_trans = TimeSeriesTransformer(input_dim=num_features, lookback=lookback).to(device)
    train_torch_model(model_trans, train_loader, epochs)
    mse, r2, pearson, thresh_trans = evaluate_regression_validation(model_trans, val_loader)
    resultados_val["Transformer"] = {"MSE": mse, "R2": r2, "Pearson": pearson}
    f1, p, r = evaluate_and_get_metrics_v2(model_trans, test_loader, labels[int(len(labels)*0.80):], lookback, thresh_trans)
    resultados_test["Transformer"] = {"F1-Score": f1, "Precision": p, "Recall": r}

    # --- Modelo 7: GCN + LSTM ---
    print("Treinando e Avaliando: GCN+LSTM...")
    model_gcn = GCN_LSTM(num_nodes=6, node_features=2, hidden_dim=16, lookback=lookback, adj_matrix=adj_matrix).to(device)
    train_torch_model(model_gcn, train_loader, epochs)
    mse, r2, pearson, thresh_gcn = evaluate_regression_validation(model_gcn, val_loader)
    resultados_val["GCN+LSTM"] = {"MSE": mse, "R2": r2, "Pearson": pearson}
    f1, p, r = evaluate_and_get_metrics_v2(model_gcn, test_loader, labels[int(len(labels)*0.80):], lookback, thresh_gcn)
    resultados_test["GCN+LSTM"] = {"F1-Score": f1, "Precision": p, "Recall": r}

    # --- Modelo 8: LSTM + XGBoost (Híbrido) ---
    print("Treinando e Avaliando: LSTM+XGBoost...")
    base_lstm = BaselineRNNs(num_features, 32, num_features, model_type='LSTM').to(device)
    train_torch_model(base_lstm, train_loader, epochs)
    
    # Extração de resíduos de treino para ajustar o XGBoost
    base_lstm.eval()
    train_preds, train_targets = [], []
    with torch.no_grad():
        for x, y in train_loader:
            train_preds.extend(base_lstm(x.to(device)).cpu().numpy())
            train_targets.extend(y.numpy())
    residuals = np.array(train_targets) - np.array(train_preds)
    
    xgb_regressor = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    xgb_regressor.fit(np.array(train_preds), residuals)
    
    # Avaliação da regressão na validação utilizando a predição refinada híbrida
    mse, r2, pearson, thresh_xgb = evaluate_regression_validation(base_lstm, val_loader, is_xgboost=True, xgboost_model=xgb_regressor)
    resultados_val["LSTM+XGBoost"] = {"MSE": mse, "R2": r2, "Pearson": pearson}
    
    f1, p, r = evaluate_and_get_metrics_v2(base_lstm, test_loader, labels[int(len(labels)*0.80):], lookback, thresh_xgb, is_xgboost=True, xgboost_model=xgb_regressor)
    resultados_test["LSTM+XGBoost"] = {"F1-Score": f1, "Precision": p, "Recall": r}

    # ==========================================
    # EXIBIÇÃO FINAL DAS TABELAS DE RESULTADOS
    # ==========================================
    print("\n" + "="*54)
    print(" 1. MÉTRICAS DE REGRESSÃO (CONJUNTO DE VALIDAÇÃO - NORMAL)")
    print("="*54)
    print(f"{'MODELO':<18} | {'MSE':<10} | {'R2-SCORE':<10} | {'PEARSON (r)':<10}")
    print("="*54)
    for m_name, metrics in resultados_val.items():
        print(f"{m_name:<18} | {metrics['MSE']:<10.5f} | {metrics['R2']:<10.4f} | {metrics['Pearson']:<10.4f}")
    print("="*54)

    print("\n" + "="*50)
    print(" 2. MÉTRICAS DE DETECÇÃO DE FALHAS (CONJUNTO DE TESTE - HARD)")
    print("="*50)
    print(f"{'MODELO':<18} | {'F1-SCORE':<10} | {'PRECISION':<10} | {'RECALL':<10}")
    print("="*50)
    for m_name, metrics in resultados_test.items():
        print(f"{m_name:<18} | {metrics['F1-Score']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f}")
    print("="*50)