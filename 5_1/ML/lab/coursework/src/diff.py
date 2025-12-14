import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----- Generate simple 1D dataset (Age only) -----
np.random.seed(42)
X = np.linspace(0, 80, 400).reshape(-1, 1)
y = (X[:, 0] > 30).astype(int)  # simple threshold rule

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- Train Random Forest -----
rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
rf.fit(X, y)
rf_pred = rf.predict_proba(X)[:, 1]

# ----- Train Surrogate NN -----
class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = Surrogate()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

for _ in range(200):
    optimizer.zero_grad()
    out = model(X_t)
    loss = loss_fn(out, y_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    nn_pred = model(X_t).numpy().ravel()

# ----- Plot Random Forest -----
plt.figure()
plt.plot(X[:, 0], rf_pred)
plt.xlabel("Age")
plt.ylabel("P(Survive)")
plt.title("Random Forest: piecewise-constant (non-differentiable)")
plt.show()

# ----- Plot Surrogate NN -----
plt.figure()
plt.plot(X[:, 0], nn_pred)
plt.xlabel("Age")
plt.ylabel("P(Survive)")
plt.title("Surrogate NN: smooth & differentiable")
plt.show()
