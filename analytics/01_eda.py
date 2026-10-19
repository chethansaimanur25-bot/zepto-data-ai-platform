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
