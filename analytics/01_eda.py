

import pandas as pd
import seaborn as sns

# The ONE AND ONLY load of the raw dataset for the whole module
df = sns.load_dataset("titanic")
df.to_csv("analytics/titanic.csv", index=False)

print("SHAPE:", df.shape)
print()
df.info()
print()
print(df.describe())
print()

miss = (df.isna().mean() * 100).round(2)
miss = miss[miss > 0].sort_values(ascending=False)
print("MISSING VALUES (% per column):")
print(miss)

# ---- Task 2: missing-value handling (threshold rule) ----
print()
print("=== MISSING-VALUE DECISIONS ===")
df_clean = df.copy()

for col, pct in miss.items():
    if pct < 5:
        action = "DROP ROWS (under 5% rule)"
    elif pct <= 30:
        action = "IMPUTE (5-30% rule)"
    else:
        action = "DROP COLUMN (over 30%, imputation unreliable)"
    print(f"{col}: {pct}% missing -> {action}")

    if pct < 5:
        df_clean = df_clean.dropna(subset=[col])
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        df_clean = df_clean.drop(columns=[col])

print()
print("cleaned shape:", df_clean.shape)
print("remaining missing cells:", int(df_clean.isna().sum().sum()))

# ---- Task 3: univariate analysis (age, fare) ----
import os
import matplotlib.pyplot as plt

os.makedirs("analytics/charts", exist_ok=True)

for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_clean[col], bins=30, edgecolor="black")
    axes[0].set_title(f"Histogram of {col}")
    axes[1].boxplot(df_clean[col], vert=False)
    axes[1].set_title(f"Box plot of {col}")
    plt.tight_layout()
    plt.savefig(f"analytics/charts/{col}_hist_box.png")
    plt.close()

print()
print("=== IQR OUTLIER COUNTS ===")
for col in ["age", "fare"]:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df_clean[col] < low) | (df_clean[col] > high)).sum())
    print(f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
          f"bounds=[{low:.2f}, {high:.2f}], outliers={n_out}")

print()
print("=== FARE: MEAN / MEDIAN / MODE ===")
f_mean = df_clean["fare"].mean()
f_median = df_clean["fare"].median()
f_mode = df_clean["fare"].mode()[0]
print(f"mean={f_mean:.2f}, median={f_median:.2f}, mode={f_mode:.2f}")
if f_mean > f_median > f_mode:
    print("fare is RIGHT-SKEWED (mean > median > mode)")
elif f_mean < f_median < f_mode:
    print("fare is LEFT-SKEWED (mean < median < mode)")
else:
    print("fare is roughly SYMMETRIC")

    # ---- Task 4: bivariate analysis ----
print()
print("=== SURVIVAL RATE BY SEX (boolean masking) ===")
for s in ["female", "male"]:
    mask = df_clean["sex"] == s
    print(f"{s}: {df_clean[mask]['survived'].mean():.4f}  (n={int(mask.sum())})")

print()
print("=== SURVIVAL RATE BY PCLASS ===")
for p in [1, 2, 3]:
    mask = df_clean["pclass"] == p
    print(f"class {p}: {df_clean[mask]['survived'].mean():.4f}  (n={int(mask.sum())})")

print()
print("=== SURVIVAL RATE BY SEX AND PCLASS ===")
for s in ["female", "male"]:
    for p in [1, 2, 3]:
        mask = (df_clean["sex"] == s) & (df_clean["pclass"] == p)
        print(f"{s}, class {p}: {df_clean[mask]['survived'].mean():.4f}  (n={int(mask.sum())})")

print()
print("=== CORRELATION MATRIX (6 columns) ===")
cols6 = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[cols6].corr()
print(corr.round(3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix (6 numeric columns)")
plt.tight_layout()
plt.savefig("analytics/charts/corr_heatmap.png")
plt.close()

pairs = []
for i, a in enumerate(cols6):
    for b in cols6[i + 1:]:
        pairs.append((a, b, corr.loc[a, b]))
pairs.sort(key=lambda t: abs(t[2]), reverse=True)

print()
print("=== TWO STRONGEST CORRELATIONS (by absolute value) ===")
for a, b, r in pairs[:2]:
    print(f"{a} vs {b}: r = {r:.3f}")

    # ---- Task 5: multivariate data story (5 charts) ----
sns.set_theme(style="whitegrid")

def save(name):
    plt.tight_layout()
    plt.savefig(f"analytics/charts/{name}.png")
    plt.close()

# Chart 1: survival by sex
plt.figure(figsize=(5, 4))
sns.barplot(data=df_clean, x="sex", y="survived", errorbar=None)
plt.title("Survival rate by sex")
save("story1_sex")

# Chart 2: survival by class and sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by class and sex")
save("story2_class_sex")

# Chart 3: age by survival outcome
plt.figure(figsize=(5, 4))
sns.boxplot(data=df_clean, x="survived", y="age")
plt.title("Age by survival (0 = died, 1 = survived)")
save("story3_age")

# Chart 4: fare by class and survival (log scale because fare is right-skewed)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="pclass", y="fare", hue="survived")
plt.yscale("log")
plt.title("Fare by class and survival (log scale)")
save("story4_fare_class")

# Chart 5: heatmap of survival by age group and sex
df_clean["age_group"] = pd.cut(
    df_clean["age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["child 0-12", "teen 13-18", "adult 19-35", "middle 36-60", "senior 60+"])
pivot = df_clean.pivot_table(index="age_group", columns="sex",
                             values="survived", aggfunc="mean", observed=True)
plt.figure(figsize=(5, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Survival rate by age group and sex")
save("story5_age_sex_heatmap")

print()
print("=== STORY SUPPORT NUMBERS ===")
print("survival by sex:")
print(df_clean.groupby("sex")["survived"].mean().round(4))
print("median age by survived:")
print(df_clean.groupby("survived")["age"].median())
print("median fare by pclass:")
print(df_clean.groupby("pclass")["fare"].median())
print("survival by age group and sex:")
print(pivot.round(3))

# ---- Task 6: exploratory standardization check ----
from sklearn.preprocessing import StandardScaler

before = df_clean[["age", "fare"]]
after = pd.DataFrame(StandardScaler().fit_transform(before),
                     columns=["age_z", "fare_z"], index=before.index)
print()
print("=== STANDARDIZATION CHECK (age, fare) ===")
print("BEFORE: mean", before.mean().round(3).to_dict(),
      "std", before.std(ddof=0).round(3).to_dict())
print("AFTER : mean", after.mean().round(6).to_dict(),
      "std", after.std(ddof=0).round(6).to_dict())