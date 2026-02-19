import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
X = np.random.uniform(0, 4*np.pi, 600).reshape(-1, 1)
y = np.sin(X) + 0.3*np.sin(5*X) + np.random.normal(0, 0.1, X.shape)

print("Data shape:", X.shape, y.shape)

# Разделите данные на обучающую, проверочную и тестовую выборки (80/10/10)
def split_data(X, y, train_size=0.8, val_size=0.1):
    total_samples = X.shape[0]
    train_end = int(total_samples * train_size)
    val_end = int(total_samples * (train_size + val_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Initialize weights and biases
def weights_init(input_size, hidden_sizes, output_size):
    weights = []
    biases = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes) - 1):
        # He initialization for ReLU
        weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
        bias_vector = np.zeros((1, layer_sizes[i+1]))
        weights.append(weight_matrix)
        biases.append(bias_vector)

    return weights, biases

# Forward pass with biases
def forward_pass(X, weights, biases):
    activations = [X]
    pre_activations = []  # Store values before ReLU for backward pass

    # Hidden layers with ReLU
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        pre_activations.append(z)
        a = relu(z)
        activations.append(a)

    # Output layer (linear)
    z_output = np.dot(activations[-1], weights[-1]) + biases[-1]
    pre_activations.append(z_output)
    activations.append(z_output)

    return activations, pre_activations

# Compute MSE loss
def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Backward pass
def backward_pass(activations, pre_activations, weights, biases, y_true):
    m = y_true.shape[0]  # number of samples
    grads_w = []
    grads_b = []

    # Output layer gradient (linear, so derivative = 1)
    dZ = (activations[-1] - y_true) / m  # scale by 1/m for MSE

    # Backpropagate through layers
    for i in range(len(weights) - 1, -1, -1):
        # Gradient for weights and biases
        dW = np.dot(activations[i].T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        grads_w.insert(0, dW)
        grads_b.insert(0, db)

        # Gradient for previous layer (if not input layer)
        if i > 0:
            dA_prev = np.dot(dZ, weights[i].T)
            dZ = dA_prev * relu_derivative(pre_activations[i-1])

    return grads_w, grads_b

# Update weights and biases
def update_parameters(weights, biases, grads_w, grads_b, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_w[i]
        biases[i] -= learning_rate * grads_b[i]

# Prediction function (returns only final output)
def predict(X, weights, biases):
    activations, _ = forward_pass(X, weights, biases)
    return activations[-1]

# Training function
def train(X_train, y_train, X_val, y_val,
          input_size=1, hidden_sizes=[64, 32], output_size=1,
          learning_rate=0.01, epochs=2000, print_every=200):

    weights, biases = weights_init(input_size, hidden_sizes, output_size)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Forward pass on training data
        activations, pre_activations = forward_pass(X_train, weights, biases)
        train_loss = compute_mse(activations[-1], y_train)
        train_losses.append(train_loss)

        # Backward pass and update
        grads_w, grads_b = backward_pass(activations, pre_activations, weights, biases, y_train)
        update_parameters(weights, biases, grads_w, grads_b, learning_rate)

        # Validation
        val_pred = predict(X_val, weights, biases)
        val_loss = compute_mse(val_pred, y_val)
        val_losses.append(val_loss)

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Train MSE: {train_loss:.6f}, Val MSE: {val_loss:.6f}')

    return weights, biases, train_losses, val_losses

# Train the model
print("\nStarting training...")
trained_weights, trained_biases, train_losses, val_losses = train(
    X_train, y_train, X_val, y_val,
    learning_rate=0.01,  # Increased learning rate
    epochs=2000,  # More epochs
    print_every=200
)

# Evaluate on test set
test_predictions = predict(X_test, trained_weights, trained_biases)
test_loss = compute_mse(test_predictions, y_test)
print(f"\nTest MSE: {test_loss:.6f}")

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Training and validation loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot 2: Predictions vs actual on test set
plt.subplot(2, 2, 2)
# Sort for better visualization
sort_idx = np.argsort(X_test.flatten())
plt.plot(X_test[sort_idx], y_test[sort_idx], 'b.', label='Actual', alpha=0.6, markersize=3)
plt.plot(X_test[sort_idx], test_predictions[sort_idx], 'r.', label='Predicted', alpha=0.6, markersize=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Test Set: Predictions vs Actual (MSE: {test_loss:.4f})')
plt.legend()
plt.grid(True)

# Plot 3: Full function approximation
plt.subplot(2, 2, 3)
# Generate smooth curve for visualization
X_smooth = np.linspace(0, 4*np.pi, 1000).reshape(-1, 1)
y_smooth_true = np.sin(X_smooth) + 0.3*np.sin(5*X_smooth)
y_smooth_pred = predict(X_smooth, trained_weights, trained_biases)

plt.plot(X_smooth, y_smooth_true, 'g-', label='True function', linewidth=2)
plt.plot(X_smooth, y_smooth_pred, 'r--', label='Model prediction', linewidth=2)
plt.scatter(X_train[::10], y_train[::10], alpha=0.3, s=5, c='blue', label='Training data (sampled)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Function Approximation')
plt.legend()
plt.grid(True)

# Plot 4: Error distribution
plt.subplot(2, 2, 4)
errors = (test_predictions - y_test).flatten()
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print model performance summary
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Final Training MSE: {train_losses[-1]:.6f}")
print(f"Final Validation MSE: {val_losses[-1]:.6f}")
print(f"Test MSE: {test_loss:.6f}")
print(f"Train-Val Gap: {abs(train_losses[-1] - val_losses[-1]):.6f}")
print("="*50)

# Additional analysis
print("\n" + "="*50)
print("MODEL ANALYSIS")
print("="*50)
print(f"Best Training MSE: {min(train_losses):.6f} at epoch {np.argmin(train_losses)}")
print(f"Best Validation MSE: {min(val_losses):.6f} at epoch {np.argmin(val_losses)}")

# Check for overfitting
if val_losses[-1] > 1.2 * train_losses[-1]:
    print("⚠️  Warning: Possible overfitting detected")
else:
    print("✅ Model appears to generalize well")
print("="*50)
