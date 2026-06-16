import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print("=== Inicializando o SVC para Classificação (UFES) ===")

# 1. Carregar os datasets (ajuste os nomes se necessário)
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
X_test = df_test[colunas_x].values

# --- ADAPTAÇÃO PARA CLASSIFICAÇÃO ---
# Caso seu Alvo_Y ainda seja contínuo, vamos binarizá-lo pela mediana para fins de teste
# Se o seu dataset já tiver classes (0 e 1), comente as duas linhas abaixo e use y = df['Alvo_Y']
mediana = np.median(df_train['Alvo_Y'].values)
y_train = np.where(df_train['Alvo_Y'].values > mediana, 1, 0)
y_test = np.where(df_test['Alvo_Y'].values > mediana, 1, 0)
# ------------------------------------

# 2. Treinar o SVC com Kernel RBF
print("-> Treinando o classificador SVC...")
classificador_svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
classificador_svc.fit(X_train, y_train)

# 3. Predição e Avaliação
y_pred = classificador_svc.predict(X_test)

print("\n=========================================================")
print("RELATÓRIO DE CLASSIFICAÇÃO DO SVC:")
print("=========================================================")
print(classification_report(y_test, y_pred, target_names=['Classe 0 (Estável)', 'Classe 1 (Instável)']))
print("=========================================================")

# 4. Matriz de Confusão Visual
print("\n-> Gerando gráfico da Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Estável (0)', 'Instável (1)'], 
            yticklabels=['Estável (0)', 'Instável (1)'])
plt.title('Matriz de Confusão - SVC', fontsize=12, fontweight='bold')
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Real do Dataset')

plt.savefig('matriz_confusao_svc.png', bbox_inches='tight', dpi=300)
print("Gráfico salvo com sucesso em 'matriz_confusao_svc.png'!")