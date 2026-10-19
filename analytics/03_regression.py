import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("analytics/titanic.csv")   # same committed CSV, no re-load

NUM = ["pclass", "age", "sibsp", "parch"]
CAT = ["sex", "embarked"]
X = df[NUM + CAT]
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])
reg = Pipeline([("prep", prep), ("model", LinearRegression())])
reg.fit(X_train, y_train)          # fit on train only
pred = reg.predict(X_test)         # transform-only on test

n = len(y_test)
p = reg.named_steps["prep"].transform(X_test).shape[1]
mae = mean_absolute_error(y_test, pred)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

metrics = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2,
                         "Adjusted_R2": adj_r2}]).round(4)
metrics.to_csv("analytics/regression_metrics.csv", index=False)
print("=== REGRESSION: predict fare ===")
print("n_test =", n, "| predictors after encoding =", p)
print(metrics.to_string(index=False))

# Residual plot + a numeric heteroscedasticity check
resid = y_test - pred
rho, pval = spearmanr(np.abs(resid), pred)
print()
print("Spearman corr between |residual| and predicted fare:",
      round(float(rho), 3), "(p =", round(float(pval), 4), ")")

plt.figure(figsize=(6, 4))
plt.scatter(pred, resid, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual plot: linear regression for fare")
plt.tight_layout()
plt.savefig("analytics/charts/regression_residuals.png")
plt.close()