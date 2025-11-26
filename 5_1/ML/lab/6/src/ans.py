import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------
# Utility: metrics
# -----------------------
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# -----------------------
# Load data (local CSV saved earlier)
# -----------------------
df = pd.read_csv("energy_efficiency.csv")
X = df.iloc[:, :-2]   # X1..X8
y = df.iloc[:, -2:]   # Y1, Y2

# reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Base model: Polynomial Regression (GLM) used in LR5
# We'll build a pipeline: PolynomialFeatures(degree=2) -> StandardScaler -> LinearRegression
# Use this pipeline as the "base" estimator when applicable.
# -----------------------
poly_degree = 2
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

# Fit base model separately per target (multi-output via MultiOutputRegressor)
# but for interpretability we also train two single-output pipelines:
base_results = {}
for target in ["Y1", "Y2"]:
    model =  base_pipeline
    model.fit(X_train, y_train[target])
    y_pred = model.predict(X_test)
    base_results[target] = evaluate(y_test[target], y_pred)

# -----------------------
# 1) Bagging (base estimator = poly linear)
# -----------------------
# Bagging does not require base estimator to be complex; use base_pipeline's steps except final linear regression can't be passed as pipeline inside BaggingRegressor
# We'll pass a regressor that itself includes poly transform: use Pipeline as base_estimator via BaggingRegressor (works in sklearn>=0.24)
bagging_grid = {
    "n_estimators": [20, 50, 100],
    "max_samples": [0.6, 0.8, 1.0],
    "max_features": [0.8, 1.0]
}

# We'll tune per-target to keep it simple and transparent
def grid_search_bagging(X_tr, y_tr, X_val, y_val):
    best = None
    best_params = None
    best_score = -np.inf
    for n in bagging_grid["n_estimators"]:
        for ms in bagging_grid["max_samples"]:
            for mf in bagging_grid["max_features"]:
                bag = BaggingRegressor(
                    estimator=Pipeline([
                        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                        ("scaler", StandardScaler()),
                        ("lr", LinearRegression())
                    ]),
                    n_estimators=n, max_samples=ms, max_features=mf, random_state=42, n_jobs=-1
                )
                bag.fit(X_tr, y_tr)
                y_pred = bag.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                if r2 > best_score:
                    best_score = r2
                    best = bag
                    best_params = {"n_estimators": n, "max_samples": ms, "max_features": mf}
    return best, best_params

bagging_results = {}
# for target in ["Y1", "Y2"]:
#     model_bag, params_bag = grid_search_bagging(X_train, y_train[target], X_test, y_test[target])
#     y_pred = model_bag.predict(X_test)
#     bagging_results[target] = {"model": model_bag, "params": params_bag, **evaluate(y_test[target], y_pred)}

# -----------------------
# 2) Boosting: HistGradientBoostingRegressor (fast, supports multioutput via wrapper)
# We'll tune learning_rate and max_iter / max_depth
# -----------------------
boost_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 300, 500],
    "max_depth": [3, 6, None]
}

boosting_results = {}
for target in ["Y1", "Y2"]:
    best = None
    best_params = None
    best_score = -np.inf
    for lr in boost_params["learning_rate"]:
        for it in boost_params["max_iter"]:
            for md in boost_params["max_depth"]:
                # HistGradientBoostingRegressor doesn't accept None for max_depth in some versions; treat None as no limit
                hgb = HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, max_depth=md, random_state=42)
                hgb.fit(X_train, y_train[target])
                y_pred = hgb.predict(X_test)
                r2 = r2_score(y_test[target], y_pred)
                if r2 > best_score:
                    best_score = r2
                    best = hgb
                    best_params = {"learning_rate": lr, "max_iter": it, "max_depth": md}
    y_pred = best.predict(X_test)
    boosting_results[target] = {"model": best, "params": best_params, **evaluate(y_test[target], y_pred)}

# -----------------------
# 3) Stacking: use a set of diverse base learners, meta-learner = LinearRegression
# We'll build stacking per-target to simplify multioutput handling.
# Base estimators: RandomForest, SVR (with RBF), Ridge linear on poly features (already strong)
# -----------------------
stacking_results = {}
from sklearn.ensemble import StackingRegressor

for target in ["Y1", "Y2"]:
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
        ("svr", Pipeline([
            ("scaler", StandardScaler()),
            ("svr_est", SVR(kernel="rbf", C=1.0, gamma="scale"))
        ])),
        ("poly_ridge", Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]))
    ]
    # final estimator
    final_est = LinearRegression()
    stack = StackingRegressor(estimators=estimators, final_estimator=final_est, n_jobs=-1, passthrough=False)
    # we can grid-search some params (but keep reasonable defaults to save time)
    stack.fit(X_train, y_train[target])
    y_pred = stack.predict(X_test)
    stacking_results[target] = {"model": stack, "params": "rf(200,d=6)+svr+poly_ridge", **evaluate(y_test[target], y_pred)}

# -----------------------
# 4) Lasso
# -----------------------

lasso_results = {}


# -----------------------
# Collect results into DataFrame for both targets and compare with base GLM
# -----------------------
rows = []
methods = [
    ("Base LM", base_results),
    ("Bagging", bagging_results),
    ("Boosting", boosting_results),
    ("Stacking", stacking_results)
]

for method_name, res_dict in methods:
    for target in ["Y1", "Y2"]:
        met = res_dict[target]
        rows.append({
            "Model": method_name,
            "Target": target,
            "MSE": met["MSE"],
            "RMSE": met["RMSE"],
            "MAE": met["MAE"],
            "R2": met["R2"],
            "Params": met.get("params", "")
        })

results_df = pd.DataFrame(rows)
# wide table: rows per model, columns for Y1/Y2
pivot = results_df.pivot_table(index="Model", columns="Target", values=["R2", "RMSE"])
print("\n=== Summary (R2 and RMSE) ===")
print(pivot)

print("\n=== Full table ===")
print(results_df.sort_values(["Target","R2"], ascending=[True, False]).to_string(index=False))
