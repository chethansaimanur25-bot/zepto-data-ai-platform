

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