import numpy as np

class MLP:

    def __init__(self, input_size, hidden1, hidden2, output_size, lr=0.02):

        self.lr = lr

        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        # Linear output
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(1 / hidden2)
        self.b3 = np.zeros((1, output_size))


    def relu(self, x):
        return np.maximum(0, x)

    def relu_d(self, x):
        return (x > 0).astype(float)


    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        return 1 - np.tanh(x) ** 2


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return self.sigmoid(x)



    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)


    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.tanh(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.Z3  # Linear

        return self.A3


    def backward(self, X, y):

        m = X.shape[0]

        y_pred = self.A3

        dZ3 = (y_pred - y) * 2 / m
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.tanh_d(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_d(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    def train(self, X, y, epochs=2000):
        self.loss_history = []

        for epoch in range(epochs):

            y_pred = self.forward(X)
            loss = self.mse(y_pred, y)

            self.loss_history.append(loss)

            self.backward(X, y)

            if epoch % 200 == 0:
                print(f"epoch {epoch}, loss {loss:.6f}")

        return self.loss_history
