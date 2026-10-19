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