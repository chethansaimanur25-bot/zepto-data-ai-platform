import pandas as pd

RATE = 105.50  # fixed project rate: 1 GBP = 105.50 INR
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

df = pd.read_csv("data_pipeline/raw_books.csv")

# price: strip symbol -> float; unparseable -> NaN -> median imputation
df["price_gbp"] = pd.to_numeric(
    df["price"].str.replace("£", "", regex=False).str.strip(),
    errors="coerce",
)
df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

# rating: text -> int 1-5; unknown text -> drop row (a median rating would be invented)
df["rating"] = df["star_rating"].map(RATING_MAP)
df = df.dropna(subset=["rating"]).copy()
df["rating"] = df["rating"].astype(int)

# availability -> boolean
df["in_stock"] = df["availability"].str.contains("in stock", case=False, na=False)

# fixed-rate conversion
df["price_inr"] = (df["price_gbp"] * RATE).round(2)

clean = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
clean.to_csv("data_pipeline/clean_books.csv", index=False)

print(clean.dtypes)
print(clean.shape)
print(clean.head())