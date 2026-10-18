# Zepto Data & AI Platform

Capstone project with three modules:

- `data_pipeline` - scrape, clean, convert and store book data, then query it (Module 1)
- `analytics` - in progress
- `support_assistant` - in progress

## Setup

One consolidated `requirements.txt` at the repo root covers all modules.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Module 1: data_pipeline

Run from the repo root, in this order:

```
python data_pipeline/scraper.py     # scrape books.toscrape.com -> raw_books.csv
python data_pipeline/clean.py       # clean + convert -> clean_books.csv
python data_pipeline/build_db.py    # create books.db (2 tables, PK/FK)
python data_pipeline/queries.py     # 6 SQL queries -> sql_output.txt
python data_pipeline/compare.py     # read_sql vs pd.merge -> pandas_output.txt
```

`books.db` is excluded by `.gitignore`; `build_db.py` regenerates it from scratch.

### Design decisions

- **Scope:** 4 categories (Travel, Mystery, Historical Fiction, Classics) = 88 books, all pagination followed.
- **Currency:** fixed project rate **1 GBP = 105.50 INR**. This is a project-defined constant, not a live market rate, so no API call is made.
- **price_gbp:** the currency symbol is stripped and the value cast to float. Unparseable prices are filled with the column median.
- **rating:** text One..Five mapped to int 1-5. Rows with unrecognised text are dropped, because inventing a median rating would distort the data.
- **in_stock:** boolean, True when the availability text contains "in stock".
- **Schema:** `categories(category_id PK, category_name UNIQUE)` and `books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)`.
- **SQL:** 6 queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN and JOINs. Results are saved in `sql_output.txt`.
- **pandas:** `pd.read_sql` results and a `pd.merge` reproduction of the join are in `pandas_output.txt`; the script confirms they are equivalent.