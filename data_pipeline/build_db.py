import sqlite3
import pandas as pd

DB = "data_pipeline/books.db"
df = pd.read_csv("data_pipeline/clean_books.csv")

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
""")

cats = sorted(df["category"].unique())
cur.executemany("INSERT INTO categories (category_name) VALUES (?)", [(c,) for c in cats])
cat_id = dict(cur.execute("SELECT category_name, category_id FROM categories").fetchall())

rows = [
    (r.title, r.price_gbp, r.price_inr, int(r.rating), int(r.in_stock), cat_id[r.category])
    for r in df.itertuples()
]
cur.executemany(
    "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?,?,?,?,?,?)",
    rows,
)
conn.commit()

print("categories:", cur.execute("SELECT COUNT(*) FROM categories").fetchone()[0])
print("books:", cur.execute("SELECT COUNT(*) FROM books").fetchone()[0])
print(cur.execute("SELECT * FROM books LIMIT 3").fetchall())
conn.close()