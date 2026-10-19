import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("analytics/titanic.csv")   # same committed CSV
NUM = ["pclass", "age", "sibsp", "parch", "fare"]
CAT = ["sex", "embarked"]
X, y = df[NUM + CAT], df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

full_pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
])
full_pipeline.fit(X_train, y_train)          # train only
joblib.dump(full_pipeline, "analytics/best_pipeline.joblib")
print("saved analytics/best_pipeline.joblib")

# Reload and check it still works on RAW input
loaded = joblib.load("analytics/best_pipeline.joblib")
pred = loaded.predict(X_test)
print("reloaded test accuracy:", round(accuracy_score(y_test, pred), 4),
      "| F1:", round(f1_score(y_test, pred), 4))

new_passengers = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0, "fare": 80.0, "embarked": "S"},
    {"pclass": 3, "sex": "male", "age": 22.0, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"},
    {"pclass": 3, "sex": "male", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 8.05, "embarked": np.nan},
])
print("raw new passengers -> predicted survived:", loaded.predict(new_passengers).tolist())
print("survival probabilities:", loaded.predict_proba(new_passengers)[:, 1].round(3).tolist())