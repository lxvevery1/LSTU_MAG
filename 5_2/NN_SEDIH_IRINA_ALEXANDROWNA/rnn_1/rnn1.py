# Predict x4 + x5 using LSTM

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Data generation
np.random.seed(42)

N = 5000
X = np.random.uniform(0, 1, (N, 10))

y = X[:, 4] + X[:, 5]

X = X.reshape(N, 10, 1)

# Model
model = Sequential([
    LSTM(16, input_shape=(10, 1)),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse']
)

# Training
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluation
pred = model.predict(X).flatten()
errors = pred - y

mse = np.mean(errors**2)
print(f"MSE: {mse:.6f}")

# Plots
# Loss
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()
plt.show()

# Scatter
plt.figure()
plt.scatter(y, pred, alpha=0.3)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Scatter")
plt.show()

# Error histogram
plt.figure()
plt.hist(errors, bins=50)
plt.title("Error distribution")
plt.show()
