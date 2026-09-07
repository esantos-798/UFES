# evaluate_multitask_monster.py
import os
import torch
import torch.nn.functional as F  # <-- IMPORT CORRIGIDO AQUI
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from src.models import MultiTaskFaultDiagnosisGNN

WINDOW_SIZE = 15
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega e prepara dados de teste
df = pd.read_csv("data/Multitask_dataset.csv")
features = [c for c in df.columns if c not in ['Timestamp', 'Label_Det', 'Label_Loc', 'Label_Sev']]
num_nodes = len(features)

X_data = df[features].values
y_det, y_loc, y_sev = df['Label_Det'].values, df['Label_Loc'].values, df['Label_Sev'].values

X_windows, seq_det, seq_loc, seq_sev = [], [], [], []
for i in range(len(df) - WINDOW_SIZE):
    X_windows.append(X_data[i:i+WINDOW_SIZE])
    seq_det.append(y_det[i+WINDOW_SIZE-1])
    seq_loc.append(y_loc[i+WINDOW_SIZE-1])
    seq_sev.append(y_sev[i+WINDOW_SIZE-1])

X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
split = int(0.8 * len(X_tensor))

# Isola apenas a porção de teste
test_X = X_tensor[split:].to(DEVICE)
true_det = np.array(seq_det[split:])
true_loc = np.array(seq_loc[split:])
true_sev = np.array(seq_sev[split:])

# Inicializa e carrega os pesos do monstro
model = MultiTaskFaultDiagnosisGNN(num_nodes=num_nodes, in_features=1, hidden_dim=64).to(DEVICE)
checkpoint = torch.load("multitask_monster_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

with torch.no_grad():
    out_det, out_loc, out_sev = model(test_X)
    
    # Aplica ativações corretas para classificação pós-logits
    pred_det = (torch.sigmoid(out_det).squeeze() > 0.5).cpu().numpy().astype(int)
    pred_loc = torch.argmax(F.softmax(out_loc, dim=1), dim=1).cpu().numpy()
    pred_sev = torch.argmax(F.softmax(out_sev, dim=1), dim=1).cpu().numpy()

print("\n" + "="*60)
print("     RELATÓRIO DE DESEMPENHO DO FRAMEWORK MULTITAREFA")
print("="*60)

print(f"\n[TAREFA 1] DETECÇÃO DE FALHA (Acurácia: {accuracy_score(true_det, pred_det):.4f})")
print(classification_report(true_det, pred_det, target_names=['Saudável', 'Em Falha'], zero_division=0))

print(f"\n[TAREFA 2] LOCALIZAÇÃO DA CAUSA RAIZ (Acurácia: {accuracy_score(true_loc, pred_loc):.4f})")
print(classification_report(true_loc, pred_loc, zero_division=0))

print(f"\n[TAREFA 3] CLASSIFICAÇÃO DE SEVERIDADE (Acurácia: {accuracy_score(true_sev, pred_sev):.4f})")
# Removido o target_names para evitar o mismatch de classes no vetor de teste
print(classification_report(true_sev, pred_sev, zero_division=0))
print("="*60)