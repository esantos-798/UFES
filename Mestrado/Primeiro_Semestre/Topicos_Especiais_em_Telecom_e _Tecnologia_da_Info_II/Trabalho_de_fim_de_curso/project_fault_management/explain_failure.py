# explain_failure.py
import os
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import LSTMBaseline

WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/SoftFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados e preparando amostras de falha...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

model_file = "lstm_soft.pt"
if not os.path.exists(model_file):
    print(f"[ERRO] Checkpoint {model_file} não encontrado.")
    exit()

model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Wrapper adaptado: recebe dados achatados do SHAP e reconstrói o formato 3D para a LSTM
def model_predict_flattened(x_flat):
    # x_flat chega como: (n_samples, WINDOW_SIZE * num_features)
    n_samples = x_flat.shape[0]
    # Reconstrói a estrutura tridimensional exigida pelo PyTorch: (n_samples, window_size, features)
    x_3d = x_flat.reshape(n_samples, WINDOW_SIZE, num_features)
    
    x_tensor = torch.tensor(x_3d, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()
    
    # Retorna o desvio médio do sinal predito
    return np.mean(preds, axis=1)

print("\n2. Inicializando o KernelExplainer com dados linearizados temporais...")
# Achatamos os históricos de fundo e de teste para 2D: (amostras, janela * features)
background_flat = X_test[:20].numpy().reshape(20, -1)

instance_idx = len(X_test) - 50
sample_to_explain_flat = X_test[instance_idx:instance_idx+1].numpy().reshape(1, -1)

explainer = shap.KernelExplainer(model_predict_flattened, background_flat)

print("3. Calculando valores SHAP bidimensionais (Aproximação Kernel)...")
shap_values = explainer.shap_values(sample_to_explain_flat, nsamples=100)

# shap_values[0] tem shape (WINDOW_SIZE * num_features,)
# Vamos reconstruir o formato para agregar o impacto por feature isolada
shap_matrix = shap_values[0].reshape(WINDOW_SIZE, num_features)

# Tira a média absoluta ao longo dos 15 passos de tempo para consolidar a importância da feature
absolute_shap = np.mean(np.abs(shap_matrix), axis=0)

total_impact = np.sum(absolute_shap)
if total_impact == 0:
    total_impact = 1e-5
impact_percentages = (absolute_shap / total_impact) * 100

df_importance = pd.DataFrame({
    'Atributo': feature_names,
    'Impacto_SHAP_Absoluto': absolute_shap,
    'Contribuicao_%': impact_percentages
}).sort_values(by='Contribuicao_%', ascending=False)

print("\n" + "="*60)
print("   EXPLICAÇÃO XAI (SHAP): POR QUE O ALARME FOI DISPARADO?")
print("="*60)
print(f"Amostra analisada (Timestep de Teste: {instance_idx})")
print("-"*60)
for idx, row in df_importance.head(5).iterrows():
    print(f"-> {row['Atributo']:<30} | Peso: {row['Impacto_SHAP_Absoluto']:.4f} | Contribuição: {row['Contribuicao_%']:.2f}%")
print("="*60)

# Gera e salva o gráfico para o relatório do Helder
plt.figure(figsize=(10, 6))
top_n = df_importance.head(8)
plt.barh(top_n['Atributo'][::-1], top_n['Contribuicao_%'][::-1], color='teal')
plt.xlabel("Contribuição para o Alarme (%)")
plt.title(f"Root Cause Analysis via SHAP (Cenário Soft Failure)")
plt.tight_layout()
plt.savefig("data/RCA_SHAP_Output.png")
print("\n[SUCESSO] Gráfico explicativo salvo em: 'data/RCA_SHAP_Output.png'")