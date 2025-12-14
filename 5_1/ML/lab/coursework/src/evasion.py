from data_preprocessing import get_processed_data
from data_preprocessing import load_models_from_file
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

X_train, X_val, y_train, y_val, df, scaler = get_processed_data(
    test_size=0.2, random_state=seed
)


scaler, dt, gnb, logreg, rf, svm = load_models_from_file()

feature_names = list(X_train.columns)
allowed_features = ["Sex", "Age", "Pclass"]
allowed_idx = [feature_names.index(f) for f in allowed_features if f in feature_names]

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

# ---------------- Evasion attack generation ----------------
def evasion_attack(X_orig, allowed_idx, magnitude=0.05):
    """
    Simple evasion attack: add random perturbations to allowed features within given magnitude.
    X_orig: np.array (n_samples, n_features) in original (unscaled) space
    allowed_idx: list of feature indices allowed to perturb
    magnitude: fraction of the feature range to perturb
    """
    X_adv = X_orig.copy()
    for idx in allowed_idx:
        # compute feature range
        f_min = X_all.iloc[:, idx].min()
        f_max = X_all.iloc[:, idx].max()
        f_range = f_max - f_min
        # random perturbation in [-magnitude*range, +magnitude*range]
        perturb = (np.random.rand(X_orig.shape[0]) * 2 - 1) * magnitude * f_range
        X_adv[:, idx] += perturb
    # clip to original bounds
    # project one-hot groups row-by-row
    for i in range(X_adv.shape[0]):
        X_adv[i] = project_onehot_row(X_adv[i], onehot_groups, feature_names)
        X_adv[i] = project_semantic_constraints(X_adv[i], feature_names)

    # clip to original observed bounds
    X_adv = np.minimum(np.maximum(X_adv, orig_min), orig_max)
    return X_adv

def project_semantic_constraints(row_arr, feature_names):
    arr = row_arr.copy()

    # Age: uint >= 0
    if "Age" in feature_names:
        i = feature_names.index("Age")
        arr[i] = int(round(arr[i]))
        arr[i] = max(arr[i], 0)

    # Pclass: uint in {1,2,3}
    if "Pclass" in feature_names:
        i = feature_names.index("Pclass")
        arr[i] = int(round(arr[i]))
        arr[i] = min(max(arr[i], 1), 3)

    # Parch: uint in [0,10]
    if "Parch" in feature_names:
        i = feature_names.index("Parch")
        arr[i] = int(round(arr[i]))
        arr[i] = min(max(arr[i], 0), 100)

    # Sex: uint in [0,1]
    if "Sex" in feature_names:
        i = feature_names.index("Sex")
        arr[i] = int(round(arr[i]))
        arr[i] = min(max(arr[i], 0), 1)
    return arr

# Run evasion attack
eps_magnitude_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]  # fraction of range
results = []

X_val_np = X_val.values  # original unscaled



for mag in eps_magnitude_list:
    X_adv = evasion_attack(X_val_np, allowed_idx, magnitude=mag)
    X_adv_df = pd.DataFrame(X_adv, columns=X_train.columns)

    # RF
    preds_rf_adv = rf.predict(X_adv_df)
    proba_rf_adv = rf.predict_proba(X_adv_df)[:, 1]
    acc_rf = accuracy_score(y_val, preds_rf_adv)
    auc_rf = roc_auc_score(y_val, proba_rf_adv)

    results.append({
        "magnitude": mag,
        "RF_acc": acc_rf,
        "RF_auc": auc_rf,
    })
    print(f"mag={mag:.3f} RF acc {acc_rf:.4f}")


# ===================== TABULAR TRACE FOR SINGLE OBJECT =====================

# prettier output: limit float precision
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== EVASION TRACE (single sample) ===")
sample_idx = 0  # first validation sample

x_orig = X_val_np[sample_idx].copy()
x_orig_s = pd.Series(x_orig, index=feature_names)

rows = []

for mag in eps_magnitude_list:
    X_adv = evasion_attack(
        X_val_np[sample_idx:sample_idx+1],
        allowed_idx,
        magnitude=mag
    )
    x_adv = X_adv[0]
    x_adv_s = pd.Series(x_adv, index=feature_names)

    row = {
        "magnitude": mag,
    }

    for f in allowed_features:
        row[f"{f}_orig"] = x_orig_s[f]
        row[f"{f}_adv"] = x_adv_s[f]
        row[f"{f}_delta"] = x_adv_s[f] - x_orig_s[f]

    rows.append(row)

trace_df = pd.DataFrame(rows)
print(trace_df)
