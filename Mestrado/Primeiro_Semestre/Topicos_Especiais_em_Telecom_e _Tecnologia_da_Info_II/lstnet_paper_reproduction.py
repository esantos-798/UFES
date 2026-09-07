"""
Reprodução fiel do LSTNet Modificado de Silva et al. (2022, IEEE TNSM)
"Learning Long- and Short-Term Temporal Patterns for ML-driven Fault
Management in Optical Communication Networks"

Principais correções em relação à versão anterior:
  1. GRU com ativação ReLU no estado candidato (célula reimplementada
     manualmente, pois nn.GRU do PyTorch usa tanh de forma fixa).
  2. Módulo recurrent-skip ADAPTATIVO (Skip-RNN, Campos et al. [28]),
     com update gate binarizado e acúmulo de potencial ũ_t (Eqs. 1-4),
     em vez do skip periódico fixo do LSTNet original.
  3. Componente autorregressivo de ordem q=2 (Eq. 6), usando apenas os
     2 últimos passos da janela, e não a janela inteira.
  4. Normalização min-max (o artigo relata explicitamente que essa
     escolha superou o z-score).
  5. Função objetivo baseada em erro absoluto (Eq. 9, SVR linear
     simplificado) -> L1Loss.
  6. Threshold de detecção definido em 99% de confiança sobre os
     Fault Indicators (FI) do treino, calculado POR PARÂMETRO
     monitorado (não um único threshold global).
  7. Hiperparâmetros exatamente como reportados na Seção V-C:
     window=5, filtros CNN=10, kernel=3, hidRNN=3, hidSkip=2, AR q=2,
     lr=0.001, batch=32, épocas=100.
  8. Protocolo de 20 trials com pesos inicializados aleatoriamente,
     reportando média ± desvio padrão (Tabela I / Fig. 5 / Fig. 6).

Pontos que permanecem como aproximação (o artigo não detalha o
suficiente para reprodução byte-a-byte):
  - A regra exata de binarização/straight-through do update gate do
    Skip-RNN não é detalhada; usamos um estimador straight-through
    padrão (hard threshold no forward, gradiente pelo valor contínuo).
  - O dataset real precisa ser baixado do repositório do InRete Lab
    (referência [44] do artigo) e deve conter as 3 features extras de
    diferença de potência entre amplificadores adjacentes.
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==========================================
# 1. CARREGAMENTO E DIVISÃO DO DATASET (70/10/20, min-max)
# ==========================================
def preparar_dados_reais(csv_path, window_size=5, batch_size=32):
    df = pd.read_csv(csv_path)

    colunas_remover = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'index' in c.lower()]
    df_limpo = df.drop(columns=colunas_remover, errors='ignore')
    df_numerico = df_limpo.select_dtypes(include=[np.number])

    # --- Extrai a coluna de rótulo de falha (se existir) ANTES de tratar
    # as features. Ela NUNCA deve entrar como variável a ser previsna
    # pelo LSTNet -- isso contaminaria a regressão e o MSE/R2/Pearson
    # médios com uma variável binária que não é uma série telemétrica.
    label_col = None
    for c in df_numerico.columns:
        if c.strip().lower() in ('failure', 'label', 'is_failure', 'fault'):
            label_col = c
            break

    raw_labels_full = None
    if label_col is not None:
        raw_labels_full = df_numerico[label_col].fillna(0).values.astype(float)
        df_numerico = df_numerico.drop(columns=[label_col])
        print(f"-> Coluna de rótulo detectada e REMOVIDA das features: '{label_col}' "
              f"(usada apenas como gabarito de avaliação)")

    df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    for col in df_numerico.columns:
        if 'ber' in col.lower() or 'fec' in col.lower():
            n_negativos = int((df_numerico[col] < 0).sum())
            if n_negativos > 0:
                print(f"-> AVISO: coluna '{col}' tem {n_negativos} valores NEGATIVOS antes "
                      f"da transformação -log10. Isso é incomum para BER/FEC e vai gerar NaN "
                      f"(log10 de número negativo é indefinido) -- verifique a unidade/escala "
                      f"dessa coluna no CSV de origem.")
            df_numerico[col] = df_numerico[col].replace(0, 1e-12)
            transformado = -np.log10(df_numerico[col])
            n_invalidos = int(transformado.isna().sum() + np.isinf(transformado).sum())
            if n_invalidos > 0:
                print(f"-> AVISO: transformação -log10 em '{col}' gerou {n_invalidos} valores "
                      f"NaN/Inf, que serão silenciosamente preenchidos por ffill/bfill logo em "
                      f"seguida. Se esse número for grande, a coluna pode estar mascarando "
                      f"dados ruins.")
            df_numerico[col] = transformado

    df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    feature_names = list(df_numerico.columns)
    features = df_numerico.values
    n_samples = len(features)

    # Divisão exata do artigo: 70% treino / 10% validação / 20% teste
    split_train = int(n_samples * 0.70)
    split_val = int(n_samples * 0.80)

    train_raw = features[:split_train]
    val_raw = features[split_train:split_val]
    test_raw = features[split_val:]

    # --- Diagnóstico: estatísticas por feature em treino/val/teste, e
    # alerta de "drift" (mudança de nível médio entre treino e teste que
    # não seja explicada só pela injeção de falha). O artigo espera que
    # o treino/validação tenham nível praticamente constante (Fig. 4).
    print("\n" + "-" * 78)
    print(f"{'Feature':<20}{'Treino μ':>12}{'Treino σ':>12}{'Teste μ':>12}{'Teste σ':>12}{'Δμ (%)':>10}")
    print("-" * 78)
    for i, name in enumerate(feature_names):
        tr_mean, tr_std = train_raw[:, i].mean(), train_raw[:, i].std()
        te_mean, te_std = test_raw[:, i].mean(), test_raw[:, i].std()
        delta_pct = 100 * abs(te_mean - tr_mean) / (abs(tr_mean) + 1e-9)
        flag = "  <-- possível drift" if delta_pct > 20 else ""
        print(f"{name:<20}{tr_mean:>12.4f}{tr_std:>12.4f}{te_mean:>12.4f}{te_std:>12.4f}{delta_pct:>9.1f}%{flag}")
    print("-" * 78 + "\n")

    test_len = len(test_raw)

    if raw_labels_full is not None:
        # Usa o rótulo REAL do dataset, já alinhado com o split de teste
        test_labels = raw_labels_full[split_val:]
        n_falhas_treino = int(np.sum(raw_labels_full[:split_val]))
        if n_falhas_treino > 0:
            print(f"-> AVISO: {n_falhas_treino} amostras de falha encontradas em "
                  f"treino/validação. O artigo assume treino/validação 100% normais; "
                  f"verifique se o split ou o dataset usado corresponde ao do InRete Lab.")
        print(f"-> Falhas reais no teste: {int(np.sum(test_labels))} de {test_len} amostras")
    else:
        # Fallback: gabarito sintético cíclico (~4 min normal / ~1 min falha,
        # amostragem ~3.5s), usado apenas se não houver coluna de rótulo real
        test_labels = np.zeros(test_len)
        amostras_normais = 69
        amostras_falha = 17
        periodo_ciclo = amostras_normais + amostras_falha
        for idx in range(test_len):
            if idx % periodo_ciclo >= amostras_normais:
                test_labels[idx] = 1.0
        print("-> Nenhuma coluna de rótulo encontrada; usando gabarito sintético cíclico "
              "(ajuste manualmente se o padrão de indução de falhas do seu dataset for diferente).")

    # Normalização MIN-MAX (conforme relatado no artigo, Seção V-C)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    train_scaled = np.nan_to_num(train_scaled, nan=0.0)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0)

    def create_sliding_windows(data, window):
        X, Y = [], []
        for i in range(len(data) - window):
            X.append(data[i: i + window])
            Y.append(data[i + window])
        return np.array(X), np.array(Y)

    X_tr, Y_tr = create_sliding_windows(train_scaled, window_size)
    X_va, Y_va = create_sliding_windows(val_scaled, window_size)
    X_te, Y_te = create_sliding_windows(test_scaled, window_size)

    test_labels_ajustado = test_labels[window_size:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)),
                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(Y_va, dtype=torch.float32)),
                             batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(Y_te, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_labels_ajustado, features.shape[1], feature_names


# ==========================================
# 2. CÉLULA GRU COM ATIVAÇÃO ReLU (o artigo usa ReLU, não tanh)
# ==========================================
class ReLUGRUCell(nn.Module):
    """GRU padrão, exceto que o estado candidato usa ReLU em vez de tanh,
    conforme reportado no artigo ("the ReLU activation function... is
    used when working with GRU units")."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        gi = self.x2h(x)
        gh = self.h2h(h)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = F.relu(i_n + r * h_n)          # <-- ReLU no lugar de tanh
        h_new = (1 - z) * n + z * h
        return h_new


class ReLUGRU(nn.Module):
    """Camada recorrente padrão (não-skip) usando ReLUGRUCell."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = ReLUGRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return h


# ==========================================
# 3. MÓDULO RECURRENT-SKIP ADAPTATIVO (Skip-RNN, Eqs. 1-4 do artigo)
# ==========================================
class AdaptiveSkipGRU(nn.Module):
    """Implementa o mecanismo de update gate binarizado do Skip-RNN
    (Campos et al. [28]), usado no artigo no lugar do skip periódico
    fixo do LSTNet original.

    u_t         = binarize(ũ_t)
    s_t         = u_t * S(s_{t-1}, v_t) + (1-u_t) * s_{t-1}
    Δũ_t        = sigmoid(W s_t + b)
    ũ_{t+1}     = u_t*Δũ_t + (1-u_t)*(ũ_t + min(Δũ_t, 1-ũ_t))
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = ReLUGRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.update_gate = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        dev = x.device
        h = torch.zeros(batch, self.hidden_size, device=dev)
        u_tilde = torch.full((batch, 1), 0.5, device=dev)

        for t in range(seq_len):
            v_t = x[:, t, :]

            # binarize com straight-through estimator (gradiente contínuo)
            u_hard = (u_tilde > 0.5).float()
            u_t = u_hard + u_tilde - u_tilde.detach()

            h_candidate = self.cell(v_t, h)
            h = u_t * h_candidate + (1 - u_t) * h

            delta_u = torch.sigmoid(self.update_gate(h))
            u_tilde = u_t * delta_u + (1 - u_t) * (u_tilde + torch.min(delta_u, 1 - u_tilde))

        return h


# ==========================================
# 4. LSTNET MODIFICADO (fiel ao artigo)
# ==========================================
class LSTNetModificado(nn.Module):
    def __init__(self, window, hidRNN, hidCNN, CNN_kernel, hidSkip, num_features, ar_order=2):
        super().__init__()
        self.window = window
        self.num_features = num_features
        self.ar_order = ar_order

        # 1. Camada convolucional (ReLU, sem pooling, conforme artigo)
        self.conv1 = nn.Conv2d(1, hidCNN, kernel_size=(CNN_kernel, num_features))

        # 2. Componente recorrente padrão (GRU com ReLU)
        self.gru = ReLUGRU(hidCNN, hidRNN)

        # 3. Componente recurrent-skip ADAPTATIVO
        self.gru_skip = AdaptiveSkipGRU(hidCNN, hidSkip)

        # Camada de regressão densa combinando os dois ramos (Eq. 5)
        self.regression = nn.Linear(hidRNN + hidSkip, num_features)

        # 4. Componente autorregressivo de ordem q=2 (Eq. 6), por variável
        assert ar_order <= window, "ordem do AR não pode exceder a janela"
        self.ar = nn.Linear(ar_order, 1)

        # 5. Camada de fusão totalmente conectada (Eqs. 7-8)
        self.output_layer = nn.Linear(num_features, num_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (batch, window, num_features)
        c = x.view(-1, 1, self.window, self.num_features)
        c = F.relu(self.conv1(c))                     # artigo usa ReLU na conv
        c = torch.squeeze(c, 3).permute(0, 2, 1)       # (batch, seq_reduzida, hidCNN)

        r = self.gru(c)                                # (batch, hidRNN)
        s = self.gru_skip(c)                            # (batch, hidSkip)

        h_D = self.regression(torch.cat((r, s), dim=1))  # Eq. 5 (h^D_t)

        # Componente AR de ordem 2, por feature -- vetorizado (mesmos pesos
        # do AR aplicados a cada feature independentemente; equivalente
        # matematicamente ao antigo laço Python, porém sem o overhead de
        # lançar num_features operações minúsculas por forward pass).
        feat_slices = x[:, -self.ar_order:, :].permute(0, 2, 1)  # (batch, num_features, ar_order)
        h_AR = self.ar(feat_slices).squeeze(-1)                  # (batch, num_features)

        h_out = h_D + h_AR                              # Eq. 7
        return F.relu(self.output_layer(h_out))          # Eq. 8


# ==========================================
# 5. TREINO (Eq. 9: erro absoluto, SVR linear simplificado)
# ==========================================
def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # Eq. 9 minimiza soma de |Y - Ŷ|

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


# ==========================================
# 6. AVALIAÇÃO DE REGRESSÃO (MSE, R2, Pearson)
# ==========================================
def evaluate_regression(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    pearsons = []
    for i in range(all_targets.shape[1]):
        if np.std(all_targets[:, i]) > 1e-6 and np.std(all_preds[:, i]) > 1e-6:
            r, _ = pearsonr(all_targets[:, i], all_preds[:, i])
            pearsons.append(r)
    pearson_avg = np.mean(pearsons) if pearsons else 0.0

    return mse, r2, pearson_avg, all_preds, all_targets


# ==========================================
# 7. FAULT INDICATORS E THRESHOLDS POR PARÂMETRO (99% de confiança)
# ==========================================
def compute_fi_per_feature(model, loader):
    """Erro absoluto por amostra e por feature (Fault Indicator)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    fi = np.abs(all_targets - all_preds)  # (n_amostras, n_features)
    return fi


def evaluate_failure_detection(model, train_loader, test_loader, true_labels, confidence=99.0):
    """Threshold linear por parâmetro, calculado sobre 'confidence'% dos
    Fault Indicators do TREINO (conforme Seção III-B / página 8)."""
    fi_train = compute_fi_per_feature(model, train_loader)
    thresholds = np.percentile(fi_train, confidence, axis=0)  # 1 threshold por feature

    fi_test = compute_fi_per_feature(model, test_loader)
    n_preds = fi_test.shape[0]
    aligned_labels = true_labels[-n_preds:]

    # Falha detectada na amostra se QUALQUER parâmetro monitorado
    # ultrapassar seu threshold individual (localização por equipamento)
    outliers_per_feature = fi_test > thresholds  # (n_amostras, n_features)
    preds_binary = outliers_per_feature.any(axis=1).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        aligned_labels, preds_binary, average='binary', zero_division=0
    )
    return f1, prec, rec, thresholds, outliers_per_feature


# ==========================================
# 8. EXECUÇÃO: 20 TRIALS (Tabela I / Fig. 5 / Fig. 6)
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprodução do LSTNet Modificado (Silva et al., 2022)")
    parser.add_argument("--trials", type=int, default=20, help="Número de trials (padrão do artigo: 20)")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas por trial (padrão do artigo: 100)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                         help="Threads de CPU para o PyTorch (padrão: todos os núcleos disponíveis)")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    print("--- Reprodução fiel do LSTNet Modificado (Silva et al., 2022) ---")
    print(f"Dispositivo: {device} | Threads CPU: {torch.get_num_threads()}")
    if args.trials != 20 or args.epochs != 100:
        print(f"AVISO: rodando com --trials={args.trials} --epochs={args.epochs} "
              f"(diferente do protocolo do artigo: 20 trials x 100 épocas). "
              f"Use os padrões para comparar com a Tabela I.")

    csv_nome = "HardFailure_dataset.csv"
    if not os.path.exists(csv_nome):
        print(f"Erro: '{csv_nome}' não encontrado. Baixe do repositório InRete Lab "
              "(https://github.com/Network-And-Services/optical-failure-dataset).")
        raise SystemExit(1)

    # Hiperparâmetros exatamente como reportados na Seção V-C
    WINDOW = 5
    HID_CNN = 10
    CNN_KERNEL = 3
    HID_RNN = 3
    HID_SKIP = 2
    AR_ORDER = 2
    LR = 0.001
    BATCH_SIZE = 32
    EPOCHS = args.epochs
    N_TRIALS = args.trials
    CONFIDENCE = 99.0

    tempo_inicio = time.time()

    train_loader, val_loader, test_loader, test_labels, num_features, feature_names = preparar_dados_reais(
        csv_path=csv_nome, window_size=WINDOW, batch_size=BATCH_SIZE
    )
    print(f"Features de telemetria: {num_features} -> {feature_names}")

    results = {'mse': [], 'r2': [], 'pearson': [], 'f1': [], 'prec': [], 'rec': []}

    for trial in range(N_TRIALS):
        trial_start = time.time()
        seed_everything(trial)
        model = LSTNetModificado(
            window=WINDOW, hidRNN=HID_RNN, hidCNN=HID_CNN, CNN_kernel=CNN_KERNEL,
            hidSkip=HID_SKIP, num_features=num_features, ar_order=AR_ORDER
        ).to(device)

        train_model(model, train_loader, epochs=EPOCHS, lr=LR)

        mse, r2, pearson, _, _ = evaluate_regression(model, test_loader)
        f1, prec, rec, thresholds, _ = evaluate_failure_detection(
            model, train_loader, test_loader, test_labels, confidence=CONFIDENCE
        )

        results['mse'].append(mse)
        results['r2'].append(r2)
        results['pearson'].append(pearson)
        results['f1'].append(f1)
        results['prec'].append(prec)
        results['rec'].append(rec)

        trial_time = time.time() - trial_start
        elapsed = time.time() - tempo_inicio
        eta = trial_time * (N_TRIALS - trial - 1)
        print(f"[Trial {trial + 1:2d}/{N_TRIALS}] MSE={mse:.5f}  R2={r2:.4f}  "
              f"Pearson={pearson:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
              f"| {trial_time:.1f}s/trial, ETA {int(eta // 60)}min")

    tempo_total = time.time() - tempo_inicio

    print("\n" + "=" * 60)
    print(f" RESULTADOS FINAIS ({N_TRIALS} trials, média ± desvio padrão)")
    print("=" * 60)
    for k in ['mse', 'r2', 'pearson', 'f1', 'prec', 'rec']:
        arr = np.array(results[k])
        print(f"{k.upper():8s}: {arr.mean():.5f} ± {arr.std():.5f}")
    print("=" * 60)
    print(f"Tempo total: {int(tempo_total // 60)} min {int(tempo_total % 60)} seg")
