import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA (4 classes, 2 points each)
# =========================
P = np.array([
    [3, 2, -1, -2, -1, -2, 1, 2],   # p1
    [1, 2,  1,  2, -1, -2, -1, -2] # p2
])

labels = np.array([0,0, 1,1, 2,2, 3,3])

class_names = {
    0: "Class 1",
    1: "Class 2",
    2: "Class 3",
    3: "Class 4"
}

# =========================
# PERCEPTRON PARAMETERS
# =========================
num_classes = 4
learning_rate = 0.1
epochs = 5

np.random.seed(17)
X = np.random.uniform(0, 4*np.pi, 600).reshape(-1, 1)
y = np.sin(X) + 0.3*np.sin(5*X) + np.random.normal(0, 0.1, X.shape)

W = np.zeros((num_classes, 2))
b = np.zeros(num_classes)

# =========================
# ACTIVATION FUNCTION
# =========================
def activation(x):
    return 1 if x >= 0 else 0

# =========================
# TRAINING FUNCTION
# =========================
def train_perceptron(P, labels):
    global W, b
    for _ in range(epochs):
        for i in range(P.shape[1]):
            x = P[:, i]
            for c in range(num_classes):
                target = 1 if labels[i] == c else 0
                net = np.dot(W[c], x) + b[c]
                output = activation(net)
                error = target - output
                W[c] += learning_rate * error * x
                b[c] += learning_rate * error

train_perceptron(P, labels)

# =========================
# DECISION REGIONS
# =========================
x_min, x_max = P[0].min() - 1, P[0].max() + 1
y_min, y_max = P[1].min() - 1, P[1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = np.zeros(xx.shape)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        outputs = [np.dot(W[c], point) + b[c] for c in range(num_classes)]
        Z[i, j] = np.argmax(outputs)

# =========================
# PLOT
# =========================
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(P[0], P[1])

# Label each point with class name
for i in range(P.shape[1]):
    plt.text(
        P[0, i] + 0.1,
        P[1, i] + 0.1,
        class_names[labels[i]]
    )

# Label regions (manually positioned)
plt.text(1.5, 1.5, "Region: Class 1")
plt.text(-1.8, 1.5, "Region: Class 2")
plt.text(-1.8, -1.8, "Region: Class 3")
plt.text(1.5, -1.8, "Region: Class 4")

plt.xlabel("p1")
plt.ylabel("p2")
plt.title("Perceptron Decision Regions")

plt.show()

# =========================
# TESTING
# =========================
print("Testing results:")
for i in range(P.shape[1]):
    x = P[:, i]
    outputs = [np.dot(W[c], x) + b[c] for c in range(num_classes)]
    predicted = np.argmax(outputs)
    print(f"Point {x} -> Predicted: {class_names[predicted]}, True: {class_names[labels[i]]}")

print("\nWeights:")
print(W)

print("\nBiases:")
print(b)
