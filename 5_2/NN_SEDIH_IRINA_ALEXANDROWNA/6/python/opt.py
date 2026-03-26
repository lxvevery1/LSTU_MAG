
# Для вашего варианта (номер варианта соответствует функции и оптимизаторам из списка) выполните следующее:

# 1. Реализуйте три указанных оптимизатора на Python (без использования готовых
# реализаций из библиотек машинного обучения). Каждый оптимизатор должен быть
# реализован в виде отдельной функции или класса, принимающих на вход целевую
# функцию, её производную, начальную точку и гиперпараметры.

# 2. Для каждого оптимизатора:
#    - Подберите скорость обучения (learning rate) так, чтобы обеспечить устойчивую сходимость к минимуму за 50-200 итераций.
#    - Для импульсных методов (mSGD, NAG) используйте коэффициент импульса β = 0.9.
#    - Для адаптивных методов (RMSprop, Adam, Nadam, AdamW) используйте стандартные параметры: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸.
#    - Для AdamW используйте коэффициент затухания весов (weight decay) λ = 0.01.

# 3. Запишите траекторию оптимизации – значения x на каждой итерации (или каждой 5-й итерации для длинных траекторий).

# 4. Постройте три графика:
#    - График 1: Сама целевая функция f(x) на интервале, содержащем начальную
#    точку и найденный минимум. На этом графике отметьте начальную точку и
#    конечные точки, достигнутые каждым оптимизатором.
#    - График 2: Траектории оптимизации (значение x от итерации) для всех трёх оптимизаторов на одном графике. По оси X – номер итерации, по оси Y – текущее значение x.
#    - График 3: Значение функции f(x) от итерации для всех трёх оптимизаторов на одном графике (сходимость по значению функции). По оси X – номер итерации, по оси Y – f(x).

# 5. Проведите сравнительный анализ, ответив на вопросы:
#    - Какой оптимизатор сошёлся быстрее всего (по числу итераций)?
#    - Какой оптимизатор нашёл лучшее (меньшее) значение функции?
#    - Наблюдались ли осцилляции или нестабильность в каком-либо методе?
#    - Как форма целевой функции (количество локальных минимумов, крутизна склонов) повлияла на поведение разных оптимизаторов?
#    - Подтвердились ли теоретические ожидания о работе каждого метода (например,
#    что импульс помогает преодолевать локальные минимумы, что адаптивные методы
#    лучше справляются с оврагами и т.д.)?

# Вариант 17
# Функция: f(x) = x⁴ - 10x² + 0.5x + 8
# Оптимизаторы: SGD with Momentum (mSGD), RMSprop, AdamW
# Начальная точка: x = -3


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 10*x**2 + 0.5*x + 8

def df(x):
    return 4*x**3 - 20*x + 0.5

_lr=0.01
_beta=0.9
_iters=150

def msgd(x0, lr=_lr, beta=_beta, iters=_iters):
    x = x0
    v = 0
    trajectory = [x]

    for _ in range(iters):
        g = df(x)
        v = beta * v + lr * g
        x = x - v
        trajectory.append(x)

    return np.array(trajectory)


def rmsprop(x0, lr=_lr, beta=_beta, eps=1e-8, iters=_iters):
    x = x0
    s = 0
    trajectory = [x]

    for _ in range(iters):
        g = df(x)
        s = beta * s + (1 - beta) * g**2
        x = x - lr * g / (np.sqrt(s) + eps)
        trajectory.append(x)

    return np.array(trajectory)


def adamw(x0, lr=_lr, beta1=_beta, beta2=0.999, eps=1e-8, wd=0.01, iters=_iters):
    x = x0
    m, v = 0, 0
    trajectory = [x]

    for t in range(1, iters + 1):
        g = df(x)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * x)

        trajectory.append(x)

    return np.array(trajectory)


# =========================
# Main
# =========================

x0 = 0.4

traj_msgd = msgd(x0)
traj_rms = rmsprop(x0)
traj_adamw = adamw(x0)

iters = np.arange(len(traj_msgd))

# =========================
# Plot 1: Function
# =========================
xs = np.linspace(-4, 4, 400)

plt.figure()
plt.plot(xs, f(xs), label="f(x)")

plt.scatter(x0, f(x0), label="Start")
plt.scatter(traj_msgd[-1], f(traj_msgd[-1]), label="mSGD end")
plt.scatter(traj_rms[-1], f(traj_rms[-1]), label="RMSprop end")
plt.scatter(traj_adamw[-1], f(traj_adamw[-1]), label="AdamW end")

plt.title("Function f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# =========================
# Plot 2: Trajectories
# =========================
plt.figure()
plt.plot(iters, traj_msgd, label="mSGD")
plt.plot(iters, traj_rms, label="RMSprop")
plt.plot(iters, traj_adamw, label="AdamW")

plt.title("Optimization trajectories")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.legend()
plt.grid()

# =========================
# Plot 3: Convergence
# =========================
plt.figure()
plt.plot(iters, f(traj_msgd), label="mSGD")
plt.plot(iters, f(traj_rms), label="RMSprop")
plt.plot(iters, f(traj_adamw), label="AdamW")

plt.title("f(x) convergence")
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# =========================
# Results
# =========================
print("Final results:")
print(f"mSGD:   x = {traj_msgd[-1]:.6f}, f(x) = {f(traj_msgd[-1]):.6f}")
print(f"RMSprop: x = {traj_rms[-1]:.6f}, f(x) = {f(traj_rms[-1]):.6f}")
print(f"AdamW:  x = {traj_adamw[-1]:.6f}, f(x) = {f(traj_adamw[-1]):.6f}")

plt.show()
