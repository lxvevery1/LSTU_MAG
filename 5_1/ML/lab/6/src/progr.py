# Изучить подходы к построению моделей машинного обучнеия с использованием ансамблевых методов
# для повышения качества модели.

# 1. Для множества данных из лр4 возьмите модель, которая использовалась в лр 5
# (далее - базовая модель) и постройте ансамблевые модели, используя технологии
# стэкинга, бустинга, бэггинга.

# **Старье**

# 1.1 Загрузите из UCI ML Repository базу данных Energy efficiency.
# http://archive.ics.uci.edu/ml/datasets/Energy+efficiency
# В этой базе представлены результаты исследования энергоэффективности
# зданий в виде отопительной (heating) и холодильной (cooling) нагрузки
# в зависимости от параметров

import pandas as pd
import numpy as np

df = pd.read_csv("energy_efficiency.csv")

X = df.iloc[:, :-2]  # first 8 columns (features)
y = df.iloc[:, -2:]  # last 2 columns (targets)

print("Dataset loaded locally!")
print(X.head())
print(y.head())

rs_lost = 4815162342
rs = 42

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rs
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# **Конец старья**

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

# Stacking

modelClf = BaggingClassifier(estimator=LogisticRegression(), n_estimators=50, random_state=rs)

modelClf.fit(X_train, y_train['Y1'])
print(modelClf.score(X_test, y_test))

# 2. Настройте гиперпараметры ансамблевых методов.

# 3. Постройте таблцы для сравнения значений адекватности моделей.
# Сравните адекватность моделей с базовой моделью и лучшей моделью из работы #5.

# 4. Сделайте вывод об ансамблевых методах.
