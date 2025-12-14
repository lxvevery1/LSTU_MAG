# FGSM transfer attack via surrogate NN (PyTorch)
from data_preprocessing import get_processed_data
from data_preprocessing import load_models_from_file
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from copy import deepcopy

# === assume earlier variables exist: X_train, X_val, y_train, y_val, scaler
# === and trained models: rf, logreg, svm (svm and logreg expect scaled input)
X_train, X_val, y_train, y_val, df, scaler = get_processed_data(
    test_size=0.2, random_state=42
)

scaler, dt, gnb, logreg, rf, svm = load_models_from_file()

# ---------------- Helper utilities ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Names and indices
feature_names = list(X_train.columns)
# Choose which features are allowed to be perturbed (continuous / realistic)
# Adjust this list if you want different targetable features.
allowed_features = ["Age", "Fare", "CabinNumber", "SibSp", "Parch"]

allowed_idx = [feature_names.index(f) for f in allowed_features if f in feature_names]
print("Allowed perturbation indices:", allowed_idx)

# Identify one-hot groups to project back to valid one-hot
onehot_groups = []
# Simple heuristic: columns that start with same prefix before '_' and are categorical dummies
prefixes = {}
for col in feature_names:
    if "_" in col:
        p = col.split("_")[0]
        prefixes.setdefault(p, []).append(col)
for p, cols in prefixes.items():
    if len(cols) > 1:
        onehot_groups.append(cols)
# filter out numeric groups accidentally included (keep likely categorical groups)
# e.g., Deck_, Embarked_
onehot_groups = [
    g for g in onehot_groups if any(c.startswith(("Deck", "Embarked")) for c in g)
]
print("One-hot groups:", onehot_groups)

# bounds for clipping in original space
X_all = pd.concat([X_train, X_val], axis=0)
orig_min = X_all.min().values
orig_max = X_all.max().values


# for rounding/validating one-hot: function that sets max index to 1 others to 0
def project_onehot_row(row_arr, group_cols, feature_names):
    arr = row_arr.copy()
    for group in group_cols:
        idxs = [feature_names.index(c) for c in group]
        if max(arr[idxs]) <= 0:
            # nothing positive -> keep original argmax (to avoid all zeros)
            # set the max among original (use current values)
            arg = int(np.argmax(arr[idxs]))
        else:
            arg = int(np.argmax(arr[idxs]))
        for j, idx in enumerate(idxs):
            arr[idx] = 1.0 if j == arg else 0.0
    return arr


# ---------------- Build surrogate NN ----------------
class SurrogateMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# Prepare tensors (we train surrogate on scaled features)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_train_scaled_np = scaler.transform(X_train)  # (n_samples, n_features)
X_val_scaled_np = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled_np, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val_scaled_np, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val.values, dtype=torch.long).to(device)

surrogate = SurrogateMLP(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)

# Train surrogate for modest epochs
epochs = 120
for epoch in range(epochs):
    surrogate.train()
    optimizer.zero_grad()
    out = surrogate(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        surrogate.eval()
        with torch.no_grad():
            logits = surrogate(X_val_t)
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_val, preds)
        print(f"Surrogate epoch {epoch+1}/{epochs} val acc: {acc:.4f}")

# Evaluate baseline performance of surrogate (optional)
surrogate.eval()
with torch.no_grad():
    logits = surrogate(X_val_t)
    pred_sur = logits.argmax(dim=1).cpu().numpy()
print("Surrogate Acc (val):", accuracy_score(y_val, pred_sur))


# ---------------- FGSM attack generation ----------------
def fgsm_attack(model, x, y, eps, allowed_idx):
    """
    x: torch tensor (batch, features) - requires_grad=True
    y: true labels tensor (batch,)
    eps: epsilon in scaled feature space
    allowed_idx: list of feature indices we allow to perturb
    """
    model.eval()
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    out = model(x_adv)
    loss = criterion(out, y)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data  # (batch, features)
    # build mask for allowed indices
    mask = torch.zeros_like(grad)
    if len(allowed_idx) > 0:
        mask[:, allowed_idx] = 1.0
    # FGSM step: sign(grad) * eps, but only on allowed features
    perturb = eps * torch.sign(grad) * mask
    x_adv = x_adv + perturb
    x_adv = x_adv.detach()
    return x_adv


# Try several epsilons (in scaled domain)
eps_list = [0.01, 0.1, 0.25, 0.5, 1.0]  # experiment with these
results = []

# Baselines on clean data for reference
# For RF and other models we need original(unscaled) X_val
X_val_orig_np = X_val.values  # original domain
# LR and SVM expect scaled
from sklearn.metrics import roc_auc_score

preds_rf_clean = rf.predict(X_val_orig_np)
proba_rf_clean = rf.predict_proba(X_val_orig_np)[:, 1]
print(
    "RF clean Acc:",
    accuracy_score(y_val, preds_rf_clean),
    "AUC:",
    roc_auc_score(y_val, proba_rf_clean),
)

for eps in eps_list:
    # generate adv in scaled space (batch all validation)
    x = X_val_t.clone().detach()
    y_t = y_val_t.clone().detach()
    x_adv_scaled = fgsm_attack(surrogate, x, y_t, eps=eps, allowed_idx=allowed_idx)
    x_adv_scaled_np = x_adv_scaled.cpu().numpy()
    # For LR/SVM evaluation, supply scaled adv directly
    preds_lr_adv = logreg.predict(x_adv_scaled_np)
    proba_lr_adv = logreg.predict_proba(x_adv_scaled_np)[:, 1]
    acc_lr = accuracy_score(y_val, preds_lr_adv)
    auc_lr = roc_auc_score(y_val, proba_lr_adv)
    preds_svm_adv = svm.predict(x_adv_scaled_np)
    proba_svm_adv = svm.predict_proba(x_adv_scaled_np)[:, 1]
    acc_svm = accuracy_score(y_val, preds_svm_adv)
    auc_svm = roc_auc_score(y_val, proba_svm_adv)

    # For tree models: inverse transform to original domain, then project one-hot and clip
    x_adv_orig = scaler.inverse_transform(
        x_adv_scaled_np
    )  # back to original feature space
    # project one-hot groups row-by-row
    for i in range(x_adv_orig.shape[0]):
        x_adv_orig[i] = project_onehot_row(x_adv_orig[i], onehot_groups, feature_names)
    # clip to original observed bounds
    x_adv_orig = np.minimum(np.maximum(x_adv_orig, orig_min), orig_max)

    x_adv_orig = pd.DataFrame(x_adv_orig, columns=X_train.columns)
    preds_rf_adv = rf.predict(x_adv_orig)
    proba_rf_adv = rf.predict_proba(x_adv_orig)[:, 1]
    acc_rf = accuracy_score(y_val, preds_rf_adv)
    auc_rf = roc_auc_score(y_val, proba_rf_adv)

    results.append(
        {
            "eps": eps,
            "LR_acc": acc_lr,
            "LR_auc": auc_lr,
            "SVM_acc": acc_svm,
            "SVM_auc": auc_svm,
            "RF_acc": acc_rf,
            "RF_auc": auc_rf,
        }
    )
    print(
        f"eps={eps:.3f}  LR acc {acc_lr:.4f} -> SVM acc {acc_svm:.4f} -> RF acc {acc_rf:.4f}"
    )

# Print table-like summary
print("\nSummary (eps, LR_acc, LR_auc, SVM_acc, SVM_auc, RF_acc, RF_auc):")
for r in results:
    print(r)
