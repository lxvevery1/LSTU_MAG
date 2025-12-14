import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from data_preprocessing import get_processed_data

# ---------------- Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_val, y_train, y_val, df, scaler = get_processed_data()
feature_names = list(X_train.columns)

# scale for surrogate
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val.values, dtype=torch.long).to(device)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---------------- Surrogate NN ----------------
class SurrogateMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)

model = SurrogateMLP(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Train ----------------
for epoch in range(75):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

# ---------------- Jacobian-based sensitivity ----------------
model.eval()
X_val_t.requires_grad = True

logits = model(X_val_t)
loss = criterion(logits, y_val_t)
loss.backward()

# |∂L/∂x|
jacobian_abs = X_val_t.grad.abs().cpu().numpy()

# mean sensitivity per feature
feature_sensitivity = jacobian_abs.mean(axis=0)

sensitivity_df = pd.DataFrame({
    "feature": feature_names,
    "jacobian_sensitivity": feature_sensitivity
}).sort_values(by="jacobian_sensitivity", ascending=False)

print("\nMost vulnerable features (Jacobian-based):")
print(sensitivity_df.head(10))
