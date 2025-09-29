import pandas as pd
import sqlite3

DB = "onlineretail.db"

def setup(conn):
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS customers (customer_id INTEGER PRIMARY KEY, country TEXT);
    CREATE TABLE IF NOT EXISTS products (stock_code TEXT PRIMARY KEY, description TEXT);
    CREATE TABLE IF NOT EXISTS invoices (invoice_no TEXT PRIMARY KEY, invoice_date TIMESTAMP, customer_id INTEGER);
    CREATE TABLE IF NOT EXISTS invoice_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_no TEXT,
        stock_code TEXT,
        quantity INTEGER,
        unit_price REAL,
        FOREIGN KEY(invoice_no) REFERENCES invoices(invoice_no),
        FOREIGN KEY(stock_code) REFERENCES products(stock_code)
    );
    """)
    conn.commit()

def load(xlsx_path, db_path=DB, batch_size=1000):
    # read file (pandas will parse Excel dates to Timestamp)
    df = pd.read_excel(xlsx_path, engine="openpyxl", dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str})

    # keep only rows with CustomerID (same behaviour as your original)
    df = df[df["CustomerID"].notna()].copy()
    # normalize InvoiceDate (works for Timestamp or string)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    # drop rows where date couldn't be parsed
    df = df[df["InvoiceDate"].notna()]

    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    setup(conn)
    cur = conn.cursor()

    to_insert_lines = []
    products = set()
    customers = set()
    invoices = set()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            InvoiceNo = str(row["InvoiceNo"])
            StockCode = str(row["StockCode"])
            Description = None if pd.isna(row["Description"]) else str(row["Description"])
            Quantity = int(row["Quantity"])
            InvoiceDate = row["InvoiceDate"].to_pydatetime()   # safe conversion
            UnitPrice = float(row["UnitPrice"])
            CustomerID = int(row["CustomerID"])
            Country = None if pd.isna(row["Country"]) else str(row["Country"])

            products.add((StockCode, Description))
            customers.add((CustomerID, Country))
            invoices.add((InvoiceNo, InvoiceDate, CustomerID))
            to_insert_lines.append((InvoiceNo, StockCode, Quantity, UnitPrice))
            inserted += 1
        except Exception:
            skipped += 1
            continue

        if len(to_insert_lines) >= batch_size:
            # flush batch
            cur.executemany("INSERT OR IGNORE INTO products (stock_code, description) VALUES (?, ?)", list(products))
            cur.executemany("INSERT OR IGNORE INTO customers (customer_id, country) VALUES (?, ?)", list(customers))
            cur.executemany("INSERT OR IGNORE INTO invoices (invoice_no, invoice_date, customer_id) VALUES (?, ?, ?)", list(invoices))
            cur.executemany("INSERT INTO invoice_lines (invoice_no, stock_code, quantity, unit_price) VALUES (?, ?, ?, ?)", to_insert_lines)
            conn.commit()
            products.clear(); customers.clear(); invoices.clear(); to_insert_lines.clear()

    # final flush
    if to_insert_lines:
        cur.executemany("INSERT OR IGNORE INTO products (stock_code, description) VALUES (?, ?)", list(products))
        cur.executemany("INSERT OR IGNORE INTO customers (customer_id, country) VALUES (?, ?)", list(customers))
        cur.executemany("INSERT OR IGNORE INTO invoices (invoice_no, invoice_date, customer_id) VALUES (?, ?, ?)", list(invoices))
        cur.executemany("INSERT INTO invoice_lines (invoice_no, stock_code, quantity, unit_price) VALUES (?, ?, ?, ?)", to_insert_lines)
        conn.commit()

    conn.close()
    print(f"Done. Inserted approx {inserted} rows, skipped {skipped} rows.")

if __name__ == "__main__":
    load("Online Retail.xlsx")
