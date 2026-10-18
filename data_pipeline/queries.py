import sqlite3

DB = "data_pipeline/books.db"
OUT = "data_pipeline/sql_output.txt"

QUERIES = {
    "Q1 SELECT + WHERE + ORDER BY + LIMIT: top 10 priciest in-stock books rated 4+": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY price_gbp DESC
        LIMIT 10
    """,
    "Q2 DISTINCT: rating values present in the data": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating
    """,
    "Q3 BETWEEN: books priced GBP 20 to 30": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp
        LIMIT 10
    """,
    "Q4 IN: 4 or 5 star books in Travel or Mystery": """
        SELECT title, rating, price_inr
        FROM books
        WHERE rating IN (4, 5)
          AND category_id IN (
              SELECT category_id FROM categories
              WHERE category_name IN ('Travel', 'Mystery'))
        ORDER BY rating DESC, title
        LIMIT 10
    """,
    "Q5 JOIN: average price and book count per category": """
        SELECT c.category_name,
               COUNT(*) AS n_books,
               ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY avg_price_gbp DESC
    """,
    "Q6 JOIN: top 3 highest-rated books per category": """
        SELECT category_name, title, rating
        FROM (
            SELECT c.category_name, b.title, b.rating,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.category_id
                       ORDER BY b.rating DESC, b.title) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        WHERE rn <= 3
        ORDER BY category_name, rn
    """,
}

conn = sqlite3.connect(DB)
cur = conn.cursor()
lines = []
for name, sql in QUERIES.items():
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    lines.append(f"=== {name} ===")
    lines.append(sql.strip())
    lines.append("-- result: " + ", ".join(cols))
    lines.extend(str(r) for r in rows)
    lines.append("")
conn.close()

text = "\n".join(lines)
print(text)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)