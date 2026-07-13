import torch
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


from src.data.dataloader import get_dataloader
from src.models.transformer_encoder import TransformerEncoder

from src.utils.checkpoint import load_encoder_weights



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("Loading data...")


train_loader, val_loader, test_loader = get_dataloader()



# ==========================
# Transformer Encoder
# ==========================


encoder = TransformerEncoder(

    input_size=12,

    d_model=64,

    nhead=4,

    num_layers=2

).to(device)


encoder = load_encoder_weights(
    encoder,
    "experiments/transformer/best_model.pt"
)



# Por enquanto:
# vamos usar o encoder congelado
# depois podemos fazer fine tuning


encoder.eval()



def extract_features(loader):

    features=[]
    labels=[]


    with torch.no_grad():

        for X,y,failure in loader:


            X=X.to(device)


            z=encoder(X)


            features.append(
                z.cpu().numpy()
            )


            labels.append(
                failure.numpy()
            )


    return (

        np.concatenate(features),

        np.concatenate(labels)

    )



print("Extracting LSTM features...")


X_train,y_train = extract_features(
    train_loader
)


X_test,y_test = extract_features(
    test_loader
)



print("Feature shape")

print(X_train.shape)

print(y_train.shape)



# ==========================
# XGBoost
# ==========================


print("\nTraining XGBoost...")


xgb = XGBClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="logloss",

    random_state=42

)



xgb.fit(

    X_train,

    y_train

)



# ==========================
# Evaluation
# ==========================


pred = xgb.predict(
    X_test
)


prob = xgb.predict_proba(
    X_test
)[:,1]



print("\n===== Transformer + XGBoost =====")


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)


print(
    "Precision:",
    precision_score(
        y_test,
        pred
    )
)


print(
    "Recall:",
    recall_score(
        y_test,
        pred
    )
)


print(
    "F1:",
    f1_score(
        y_test,
        pred
    )
)


print(
    "AUC:",
    roc_auc_score(
        y_test,
        prob
    )
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        pred
    )
)