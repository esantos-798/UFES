import time, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from models import StackedRNNRegressor, LSTNet

torch.manual_seed(0)
DEVICE = "cpu"

DATA_PATH = "/home/claude/data/optical_features.csv"
WINDOW = 5
if len(sys.argv) > 3 and sys.argv[3] == "single":
    N_TRIALS = 1
    N_EPOCHS = 40
else:
    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("ts", "Failure")]
m = len(feature_cols)
Y = df[feature_cols].values.astype(np.float32)
failure_flag = df["Failure"].values.astype(int)
n = len(Y)

train_end = int(n * 0.7)
val_end = int(n * 0.8)


def make_windows(data, h):
    X, Yt = [], []
    for i in range(h, len(data)):
        X.append(data[i - h:i])
        Yt.append(data[i])
    return np.array(X, dtype=np.float32), np.array(Yt, dtype=np.float32)


def zscore_fit(train):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def minmax_fit(train):
    mn = train.min(axis=0)
    mx = train.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return mn, rng


def run_trial(model_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name == "lstnet":
        mu, rng = minmax_fit(Y[:train_end])
        Yn = (Y - mu) / rng
    else:
        mu, sd = zscore_fit(Y[:train_end])
        Yn = (Y - mu) / sd

    Xw, Yw = make_windows(Yn, WINDOW)
    flagw = failure_flag[WINDOW:]

    # window index offsets align with original index i (target at position i)
    tr_end_w = train_end - WINDOW
    val_end_w = val_end - WINDOW

    Xtr, Ytr = Xw[:tr_end_w], Yw[:tr_end_w]
    Xval, Yval = Xw[tr_end_w:val_end_w], Yw[tr_end_w:val_end_w]
    Xte, Yte = Xw[val_end_w:], Yw[val_end_w:]
    flag_te = flagw[val_end_w:]

    Xtr_t = torch.tensor(Xtr); Ytr_t = torch.tensor(Ytr)
    Xval_t = torch.tensor(Xval); Yval_t = torch.tensor(Yval)
    Xte_t = torch.tensor(Xte); Yte_t = torch.tensor(Yte)

    if model_name == "rnn":
        model = StackedRNNRegressor(m, cell="rnn")
        lr, batch = 1e-4, 128
    elif model_name == "lstm":
        model = StackedRNNRegressor(m, cell="lstm")
        lr, batch = 1e-4, 128
    elif model_name == "lstnet":
        model = LSTNet(m, window=WINDOW)
        lr, batch = 1e-3, 32
    else:
        raise ValueError(model_name)

    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = Xtr_t.shape[0]
    train_losses, val_losses = [], []

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            vpred = model(Xval_t)
            vloss = loss_fn(vpred, Yval_t).item()
        val_losses.append(vloss)

    model.eval()
    with torch.no_grad():
        pred_tr = model(Xtr_t).numpy()
        pred_val = model(Xval_t).numpy()
        pred_te = model(Xte_t).numpy()

    # de-normalize for reporting MSE in original units (paper reports raw-scale MSE)
    if model_name == "lstnet":
        def denorm(a):
            return a * rng + mu
    else:
        def denorm(a):
            return a * sd + mu

    def metrics(pred, true):
        pred_d = denorm(pred)
        true_d = denorm(true)
        mse = float(np.mean((pred_d - true_d) ** 2))
        corrs = []
        for j in range(m):
            if np.std(true_d[:, j]) < 1e-8 or np.std(pred_d[:, j]) < 1e-8:
                continue
            c = pearsonr(pred_d[:, j], true_d[:, j])[0]
            if not np.isnan(c):
                corrs.append(c)
        corr = float(np.mean(corrs)) if corrs else float("nan")
        r2 = float(r2_score(true_d, pred_d, multioutput="variance_weighted"))
        return mse, corr, r2

    mse_tr, corr_tr, r2_tr = metrics(pred_tr, Ytr)
    mse_val, corr_val, r2_val = metrics(pred_val, Yval)
    mse_te, corr_te, r2_te = metrics(pred_te, Yte)

    # Fault Indicators: per-feature squared error (proxy for euclidean distance per feature)
    fi_train = (pred_tr - Ytr) ** 2  # normalized scale, per feature
    thresholds = np.quantile(fi_train, 0.99, axis=0)

    fi_test = (pred_te - Yte) ** 2
    alarms = (fi_test > thresholds[None, :]).any(axis=1).astype(int)

    # Type I / II errors
    type1 = int(np.sum((alarms == 1) & (flag_te == 0)))
    type2 = int(np.sum((alarms == 0) & (flag_te == 1)))
    n_fail = int(flag_te.sum())
    n_normal = int((flag_te == 0).sum())

    return {
        "model": model_name, "seed": seed,
        "mse_train": mse_tr, "mse_val": mse_val, "mse_test": mse_te,
        "corr_train": corr_tr, "corr_val": corr_val, "corr_test": corr_te,
        "r2_train": r2_tr, "r2_val": r2_val, "r2_test": r2_te,
        "type1_errors": type1, "type2_errors": type2,
        "n_fail_test": n_fail, "n_normal_test": n_normal,
        "final_train_loss": train_losses[-1], "final_val_loss": val_losses[-1],
    }


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "single":
        # single mode: model_name trial_seed N_EPOCHS single
        model_name = sys.argv[1]
        seed = int(sys.argv[2])
        n_epochs_override = int(sys.argv[4]) if len(sys.argv) > 4 else N_EPOCHS
        N_EPOCHS = n_epochs_override
        t0 = time.time()
        res = run_trial(model_name, seed=seed)
        res["time_sec"] = time.time() - t0
        import os
        results_file = "/home/claude/data/results.json"
        if os.path.exists(results_file):
            with open(results_file) as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results.append(res)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(json.dumps(res))
    else:
        all_results = []
        for model_name in ["rnn", "lstm", "lstnet"]:
            for trial in range(N_TRIALS):
                t0 = time.time()
                res = run_trial(model_name, seed=trial)
                dt = time.time() - t0
                res["time_sec"] = dt
                print(json.dumps(res))
                all_results.append(res)

        with open("/home/claude/data/results.json", "w") as f:
            json.dump(all_results, f, indent=2)
