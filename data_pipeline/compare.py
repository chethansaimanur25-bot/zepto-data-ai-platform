import sqlite3
import pandas as pd

DB = "data_pipeline/books.db"
OUT = "data_pipeline/pandas_output.txt"
conn = sqlite3.connect(DB)
out = []

# Part 1: two earlier SQL results read into DataFrames with pd.read_sql
q_avg = """
    SELECT c.category_name, COUNT(*) AS n_books,
           ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    GROUP BY c.category_name
    ORDER BY avg_price_gbp DESC
"""
q_between = """
    SELECT title, price_gbp FROM books
    WHERE price_gbp BETWEEN 20 AND 30
    ORDER BY price_gbp, title LIMIT 10
"""
df_avg = pd.read_sql(q_avg, conn)
df_between = pd.read_sql(q_between, conn)
out += ["=== read_sql: avg price per category ===", df_avg.to_string(index=False), ""]
out += ["=== read_sql: books priced 20-30 GBP ===", df_between.to_string(index=False), ""]

# Part 2: the same JOIN done two ways
q_join = """
    SELECT b.title, c.category_name, b.rating, b.price_gbp
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating = 5
    ORDER BY c.category_name, b.title
"""
sql_result = pd.read_sql(q_join, conn)

books = pd.read_sql("SELECT * FROM books", conn)
cats = pd.read_sql("SELECT * FROM categories", conn)
merge_result = (
    books.merge(cats, on="category_id", how="inner")
    .query("rating == 5")[["title", "category_name", "rating", "price_gbp"]]
    .sort_values(["category_name", "title"])
    .reset_index(drop=True)
)
conn.close()

out += ["=== JOIN via SQL (pd.read_sql) ===", sql_result.head(10).to_string(index=False), ""]
out += ["=== JOIN via pd.merge (no SQL) ===", merge_result.head(10).to_string(index=False), ""]
match = sql_result.equals(merge_result)
out.append(f"rows (sql, merge): {len(sql_result)}, {len(merge_result)}")
out.append(f"Outputs equivalent: {match}")

text = "\n".join(out)
print(text)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)