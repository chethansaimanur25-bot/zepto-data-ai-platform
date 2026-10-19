import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

# Works from the committed CSV: no second sns.load_dataset call
df = pd.read_csv("analytics/titanic.csv")

NUM = ["pclass", "age", "sibsp", "parch", "fare"]
CAT = ["sex", "embarked"]
X = df[NUM + CAT]
y = df["survived"]

print("CLASS BALANCE (full data):")
print(y.value_counts(normalize=True).round(4))

# Stratified split BEFORE any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train rows:", len(X_train), "| test rows:", len(X_test))
print("train survived share:", round(y_train.mean(), 4),
      "| test survived share:", round(y_test.mean(), 4))


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

rows, fitted, rocs, cms = [], {}, {}, {}
for name, model in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    pipe.fit(X_train, y_train)            # fit on TRAIN only
    pred = pipe.predict(X_test)           # transform-only on test
    proba = pipe.predict_proba(X_test)[:, 1]
    fitted[name] = pipe
    cms[name] = confusion_matrix(y_test, pred)
    rocs[name] = roc_curve(y_test, proba)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
    })

table = pd.DataFrame(rows).round(4)
table.to_csv("analytics/classifier_metrics.csv", index=False)
print()
print("=== CLASSIFIER COMPARISON (test set) ===")
print(table.to_string(index=False))
print()
for name, cm in cms.items():
    print(name, "confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["died", "survived"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("analytics/charts/confusion_matrices.png")
plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in rocs.items():
    auc = table.loc[table["model"] == name, "auc"].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("analytics/charts/roc_curves.png")
plt.close()

# Decision tree with labelled features and classes
dt_pipe = fitted["Decision Tree"]
names = dt_pipe.named_steps["prep"].get_feature_names_out()
plt.figure(figsize=(18, 9))
plot_tree(dt_pipe.named_steps["model"], feature_names=list(names),
          class_names=["died", "survived"], filled=True, rounded=True, fontsize=8)
plt.title("Decision tree (max_depth=4)")
plt.tight_layout()
plt.savefig("analytics/charts/decision_tree.png")
plt.close()

# ---- Task 9: imbalance handling comparison (Logistic Regression) ----
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

print()
print("=== IMBALANCE HANDLING (Logistic Regression) ===")
print("train class counts:", y_train.value_counts().to_dict())

variants = {
    "baseline": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))]),
    "class_weight=balanced": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=42))]),
    # SMOTE sits inside the pipeline, so it only ever resamples the training data
    "SMOTE (train only)": ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))]),
}

imb_rows = []
for name, pipe in variants.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    imb_rows.append({
        "variant": name,
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })

imb_table = pd.DataFrame(imb_rows).round(4)
imb_table.to_csv("analytics/imbalance_comparison.csv", index=False)
print(imb_table.to_string(index=False))

# ---- Task 10: GridSearchCV on Random Forest (oob_score=True) ----
from sklearn.model_selection import GridSearchCV

print()
print("=== GRIDSEARCHCV: RANDOM FOREST ===")
rf_pipe = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestClassifier(oob_score=True, random_state=42)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7, None],
    "model__max_features": ["sqrt", "log2", 0.5],
}
gs = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)

best = gs.best_estimator_
print("best params:", gs.best_params_)
print("best CV F1:", round(gs.best_score_, 4))
print("OOB score of best model:", round(best.named_steps["model"].oob_score_, 4))

pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
tuned = pd.DataFrame([{
    "model": "Random Forest (tuned)",
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "auc": roc_auc_score(y_test, proba),
}]).round(4)
tuned.to_csv("analytics/tuned_rf_metrics.csv", index=False)
print(tuned.to_string(index=False))