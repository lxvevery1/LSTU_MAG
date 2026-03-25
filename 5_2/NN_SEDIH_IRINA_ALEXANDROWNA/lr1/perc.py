# Класс MLP с поддержкой:
#  - Различных функций активации для скрытых слоев: ReLU, Sigmoid, Tanh, Linear
#  - Выходной слой: Linear (так как используем MSE)
#  - Функция потерь: MSE (для классификации и регрессии)
#  - Метод forward, backward, train
#  - Инициализация весов (Xavier/He)

# Формат целевых значений:
# - Для регрессии: y – непрерывные значения (скаляры или векторы)
# --- Для классификации: y_onehot – one-hot encoding (например, 3 класса → [1,0,0], [0,1,0], [0,0,1])

# Вариант 17: Регрессия (сумма синусов, ReLU)

# Указания по выполнению:
# 1. Запустите код генерации данных из вашего варианта.
# 2. Разделите данные на обучающую, проверочную и тестовую выборки (80/10/10).
# 3. Реализуйте класс MLP с поддержкой:
#  - Инициализации весов (Xavier для Tanh/Sigmoid, He для ReLU)
#  - Прямого прохода с вычислением MSE
#  - Обратного прохода
#  - Обновления весов (метод обратного распространения ошибки)
# 4. Обучите сеть, сохраняя историю ошибок.
# 5. Визуализируйте:
#  - График Loss по эпохам
#  - график «предсказание vs истина»
#  - график аппроксимации
# 6. Сделайте выводы о том, как функция активации влияет на способность сети обучаться при использовании MSE для классификации.
# 7. Обратите внимание на скорость сходимости и итоговую точность.

# Задача: Регрессия (MSE)
# Активация скрытых слоев: ReLU
# Архитектура: Вход(1) -> 64 -> 32 -> Выход(1, Linear)

import numpy as np

def norm(X):
    X_mean = X.mean()
    X_std = X.std()
    return (X - X_mean) / X_std

# 1. Запустите код генерации данных из вашего варианта.
seed = 17
np.random.seed(seed)

X = np.random.uniform(0, 4*np.pi, 600).reshape(-1, 1)
y = np.sin(X) + 0.3*np.sin(5*X) + np.random.normal(0, 0.1, X.shape)

# для true графика (в исходном пространстве!)
X_sorted = np.sort(X, axis=0)
y_true = np.sin(X_sorted) + 0.3*np.sin(5*X_sorted)

from graphic import graphics
g = graphics()

# график ДО нормализации
g.draw_datagraph(X, y, X_sorted, y_true)

# 2. Разделите данные на обучающую, проверочную и тестовую выборки (80/10/10).
from sklearn.model_selection import train_test_split

random_state = 17

# 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# temp → 10% val, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=random_state
)

X_mean = X_train.mean()
X_std = X_train.std()

X_train = (X_train - X_mean) / X_std
X_val   = (X_val   - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

# 3. Реализуйте класс MLP
from mlp import MLP

# 4. Обучите сеть, сохраняя историю ошибок.
model = MLP(1, 64, 32, 1, lr=0.01)
loss_history = model.train(X_train, y_train, epochs=800)

# Проверка на test
y_pred = model.forward(X_test)

test_loss = model.mse(y_pred, y_test)

print("Test MSE:", test_loss)

# 4. Визуализация
g.draw_loss(loss_history)

X_plot = np.linspace(0, 4*np.pi, 400).reshape(-1,1)

# нормализуем для модели
X_plot_norm = (X_plot - X_mean) / X_std

y_pred = model.forward(X_plot_norm)
y_true = np.sin(X_plot) + 0.3*np.sin(5*X_plot)

# передаём ИСХОДНЫЙ X (не нормализованный)
g.def_prediction(X_plot, y_true, X, y, y_pred)
