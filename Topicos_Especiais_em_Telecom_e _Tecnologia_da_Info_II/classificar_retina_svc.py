import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

print("=== Processando Retinografia e Vetorização para SVC (Atualizado) ===")

# 1. Carregar a imagem real enviada
img = cv2.imread('image_0.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("[ERRO] Arquivo 'image_0.jpg' não encontrado!")
    exit()

# 2. Redimensionar para formato compacto
img_redimensionada = cv2.resize(img, (64, 64))

# 3. Vetorização e Normalização
vetor_imagem_real = img_redimensionada.flatten() / 255.0
print(f"-> Imagem vetorizada com sucesso! Dimensões do vetor: {vetor_imagem_real.shape}")

# 4. Dataset Simulado Balanceado
np.random.seed(42)
num_amostras_treino = 40
tamanho_vetor = 4096

# Gerando base sintética estruturada
X_train = np.array([vetor_imagem_real + np.random.normal(0, 0.08, tamanho_vetor) for _ in range(num_amostras_treino)])
# Garantindo balanceamento exato de classes
y_train = np.array([0, 1] * (num_amostras_treino // 2)) 

# 5. Treinar o SVC com a nova API de Calibração (Sem Warnings)
print("-> Treinando o SVC Calibrado...")
base_svc = SVC(kernel='rbf', C=10.0, gamma='scale')
modelo_calibrado = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
modelo_calibrado.fit(X_train, y_train)

# 6. Classificar a imagem teste
vetor_teste = vetor_imagem_real.reshape(1, -1)
classe_predita = modelo_calibrado.predict(vetor_teste)[0]
probabilidades = modelo_calibrado.predict_proba(vetor_teste)[0]

print("\n=========================================================")
print("RESULTADO DA CLASSIFICAÇÃO DA IMAGEM REGISTRADA:")
print("=========================================================")
if classe_predita == 1:
    print(f"Resultado: CLASSE 1 (Presença de Padrões de Alterações/Lesões)")
else:
    print(f"Resultado: CLASSE 0 (Padrão de Retina Saudável)")

print(f"Confiança Corrigida do Modelo: {probabilidades[classe_predita]*100:.2f}%")
print("=========================================================")