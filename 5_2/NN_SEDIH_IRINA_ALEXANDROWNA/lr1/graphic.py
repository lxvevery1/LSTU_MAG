import matplotlib.pyplot as plt

class graphics:
    def draw_datagraph(self, X, y, X_sorted, y_true):
        plt.figure(figsize=(12, 6))
        plt.scatter(X, y, alpha=0.6, s=10, label='Data')

        plt.plot(X_sorted, y_true, 'r-', linewidth=2, label='sin(X) + 0.3*sin(5X)')

        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Graph')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def draw_loss(self, loss_history):
        plt.figure(figsize=(10, 6))

        plt.plot(loss_history, linewidth=2)
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.grid()
        plt.show()

    def def_prediction(self, X_plot, y_true, X, y, y_pred):
        plt.figure(figsize=(10,6))

# исходные точки
        plt.scatter(X, y, s=10, alpha=0.3, label="Data")

# истинная функция
        plt.plot(X_plot, y_true, linewidth=3, label="True function")

# предсказание сети
        plt.plot(X_plot, y_pred, linewidth=3, label="MLP prediction")

        plt.xlabel("X")
        plt.ylabel("y")

        plt.title("Prediction vs True Function")

        plt.legend()
        plt.grid(True)

        plt.show()
