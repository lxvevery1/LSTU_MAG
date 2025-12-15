import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score

from data_preprocessing import get_processed_data, load_models_from_file

# ============================================================
# Reproducibility
# ============================================================
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load data and models
# ============================================================
X_train, X_val, y_train, y_val, df, scaler = get_processed_data(
    test_size=0.2, random_state=seed
)

scaler, dt, gnb, logreg, rf, svm = load_models_from_file()

feature_names = list(X_train.columns)

# ============================================================
# Threat model
# ============================================================
allowed_features = ["Sex", "Age", "Pclass"]
allowed_idx = [feature_names.index(f) for f in allowed_features]

print("Allowed adversarial features:", allowed_features)

# ============================================================
# Surrogate neural network
# ============================================================
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

# ============================================================
# Train surrogate on Random Forest
# ============================================================
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

rf_train_proba = rf.predict_proba(X_train.values)
rf_val_proba = rf.predict_proba(X_val.values)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

y_train_rf = torch.tensor(rf_train_proba, dtype=torch.float32).to(device)
y_val_rf = torch.tensor(rf_val_proba, dtype=torch.float32).to(device)

surrogate = SurrogateMLP(X_train.shape[1]).to(device)
optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)
criterion_mse = nn.MSELoss()

epochs = 100
print("\nTraining surrogate NN...")
for ep in range(epochs):
    surrogate.train()
    optimizer.zero_grad()

    out = torch.softmax(surrogate(X_train_t), dim=1)
    loss = criterion_mse(out, y_train_rf)
    loss.backward()
    optimizer.step()

    if (ep + 1) % 20 == 0:
        surrogate.eval()
        with torch.no_grad():
            val_out = torch.softmax(surrogate(X_val_t), dim=1)
            val_loss = criterion_mse(val_out, y_val_rf)
        print(f"Epoch {ep+1:3d}: surrogate val MSE = {val_loss.item():.6f}")

# ============================================================
# Jacobian-based vulnerability analysis
# ============================================================
surrogate.eval()
X_val_jac = X_val_t.clone().detach().requires_grad_(True)

criterion_ce = nn.CrossEntropyLoss()
loss = criterion_ce(
    surrogate(X_val_jac),
    torch.tensor(y_val.values, dtype=torch.long).to(device),
)
loss.backward()

jacobian = X_val_jac.grad.abs().mean(dim=0).cpu().numpy()

vuln_df = pd.DataFrame({
    "feature": feature_names,
    "jacobian_sensitivity": jacobian
}).sort_values("jacobian_sensitivity", ascending=False)

print("\n=== Most vulnerable features (Jacobian-based) ===")
print(vuln_df.head(10))

import pandas as pd
import matplotlib.pyplot as plt
# assume vuln_df already exists in the environment
top_k = 10
top_vuln = vuln_df.head(top_k)

plt.figure()
plt.barh(top_vuln["feature"], top_vuln["jacobian_sensitivity"])
plt.gca().invert_yaxis()
plt.xlabel("Mean absolute Jacobian value")
plt.ylabel("Feature")
plt.title("Top-10 most vulnerable features (Jacobian-based)")
plt.tight_layout()
plt.show()

# ============================================================
# Adversarial evasion attack (FGSM-style, transfer)
# ============================================================
def adversarial_evasion(x_scaled, y_true, eps, allowed_idx):
    x = x_scaled.clone().detach().to(device)
    x.requires_grad = True

    logits = surrogate(x)
    loss = criterion_ce(logits, y_true)
    loss.backward()

    grad = x.grad
    mask = torch.zeros_like(grad)
    mask[:, allowed_idx] = 1.0

    x_adv = x + eps * torch.sign(grad)
    return x_adv.detach()

# ============================================================
# Adversarial evasion attack (Jacobian-guided, targeted)
# ============================================================
def targeted_adversarial_evasion(x_scaled, y_true, eps, allowed_idx):
    """
    x_scaled: tensor (n_samples, n_features)
    y_true: tensor (n_samples,) с правильными классами
    eps: масштаб возмущения
    allowed_idx: индексы признаков для изменения
    """
    x_adv = x_scaled.clone().detach().to(device)
    x_adv.requires_grad = True

    # вычисляем логиты и целевую функцию потерь
    logits = surrogate(x_adv)
    # хотим изменить правильно классифицированные объекты
    loss = criterion_ce(logits, y_true)
    loss.backward()

    # градиенты
    grad = x_adv.grad.data

    # маска по разрешённым признакам
    mask = torch.zeros_like(grad)
    mask[:, allowed_idx] = 1.0

    # FGSM: движемся в направлении увеличения потерь
    perturb = eps * torch.sign(grad) * mask
    x_adv = x_adv + perturb

    return x_adv.detach()
# ============================================================
# Evaluate attack on Random Forest
# ============================================================
orig_min = X_train.min().values
orig_max = X_train.max().values

eps_list = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

y_val_t = torch.tensor(y_val.values, dtype=torch.long).to(device)

print("\n=== Adversarial evasion attack results (RF) ===")
results = []

for eps in eps_list:
    x_adv_scaled = targeted_adversarial_evasion(X_val_t, y_val_t, eps, allowed_idx)
    x_adv_orig = scaler.inverse_transform(x_adv_scaled.cpu().numpy())

    # clip to valid domain
    x_adv_orig = np.minimum(np.maximum(x_adv_orig, orig_min), orig_max)
    x_adv_df = pd.DataFrame(x_adv_orig, columns=feature_names)

    preds = rf.predict(x_adv_df)
    proba = rf.predict_proba(x_adv_df)[:, 1]

    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, proba)

    results.append({
        "eps": eps,
        "RF_acc": acc,
        "RF_auc": auc,
    })

    print(f"eps={eps:.3f} | RF acc={acc:.4f} | RF auc={auc:.4f}")

# ============================================================
# Final summary table
# ============================================================
print("\n=== Summary table ===")
print(pd.DataFrame(results))


import pandas as pd
import numpy as np

sample_idx = 0  # первая запись
x_val_np = X_val.values

x_orig = x_val_np[sample_idx].copy()
x_orig_s = pd.Series(x_orig, index=feature_names)

# Сохраняем трассу изменений признаков по eps
trace_rows = []

for eps in eps_list:
    x_adv_scaled = targeted_adversarial_evasion(
        X_val_t[sample_idx:sample_idx+1],
        y_val_t[sample_idx:sample_idx+1],
        eps,
        allowed_idx
    )
    x_adv_orig = scaler.inverse_transform(x_adv_scaled.cpu().numpy())
    # clip по допустимым границам
    x_adv_orig = np.minimum(np.maximum(x_adv_orig, orig_min), orig_max)
    x_adv_s = pd.Series(x_adv_orig[0], index=feature_names)

    row = {"eps": eps}
    for f in allowed_features:
        row[f"{f}_value"] = x_adv_s[f]
    trace_rows.append(row)

trace_df = pd.DataFrame(trace_rows)


import matplotlib.pyplot as plt
import pandas as pd

# превратим список словарей в DataFrame
results_df = pd.DataFrame(results)

plt.figure(figsize=(8, 4))
plt.plot(results_df["eps"], results_df["RF_acc"], marker='o', linestyle='-')
plt.xlabel("Epsilon (perturbation scale)")
plt.ylabel("Random Forest Accuracy")
plt.title("Effect of adversarial perturbation on RF accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
