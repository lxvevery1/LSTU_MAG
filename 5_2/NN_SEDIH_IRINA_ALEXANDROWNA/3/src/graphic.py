import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(0, 4*np.pi, 600).reshape(-1, 1)
y = np.sin(X) + 0.3*np.sin(5*X) + np.random.normal(0, 0.1, X.shape)

# Создание графика
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.6, s=10, label='Data')

X_sorted = np.sort(X, axis=0)
y_true = np.sin(X_sorted) + 0.3*np.sin(5*X_sorted)
plt.plot(X_sorted, y_true, 'r-', linewidth=2, label='sin(X) + 0.3*sin(5X)')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Graph')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
