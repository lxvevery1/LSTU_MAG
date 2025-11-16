# Изучить подходы к построению моделей машинного обучнеия с применением методов
# регуляризации для борьбы с переобучением.

# https://www.geeksforgeeks.org/machine-learning/lasso-vs-ridge-vs-elastic-net-ml/

# 1. Для множества даных из л.р. 4 для одной из моделей (далее - базовая модель)
# постройте модели с регуляризацией (L1-регуляризация (LASSO); L2-регуляризация
# (Ridge); комбинированная регуляризация (ElastiocNet)).

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


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# **Конец старья**

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(random_state=42),
    "Ridge": Ridge(random_state=42),
    "ElasticNet": ElasticNet(random_state=42)
}

for target in ["Y1", "Y2"]:
    print(f"\n=== {target} ===")
    for name, model in models.items():
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        print(f"{name:17s} | MSE: {mse:10.3f} | R²: {r2:.3f}")

# 2. Настройте гиперпараметры регуляризации.
print("\n")
step = 10
models = {

    "Base": (LinearRegression(), {}),

    "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 3, step)}),

    "Lasso": (Lasso(max_iter=10000), {"model__alpha": np.logspace(-3, 3, step)}),

    "ElasticNet": (
        ElasticNet(max_iter=10000),
        {
            "model__alpha": np.logspace(-3, 3, step),
            "model__l1_ratio": np.linspace(0.1, 0.9, 9)
        },
    ),
}


# 3. Постройте таблицы для сравнения значений адекватности моделей. Сравните
# адекватеонсть моделй с базоваой моделью
results = []

target_columns = ['Y1', 'Y2']

for target in target_columns:
    for name, (model, params) in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        grid = GridSearchCV(pipe, params, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train[target])

        y_pred = grid.predict(X_test)

        mse = mean_squared_error(y_test[target], y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)

        results.append({
            "Target": target,
            "Model": name,
            "Best Params": grid.best_params_,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
