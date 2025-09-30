import sqlite3
import time
from typing import Dict, Optional
from pymongo import MongoClient, errors
from pymongo import UpdateOne
import pandas as pd

# Config (tweak as needed)
DB_PATH = "onlineretail1.db"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "onlineretail"
BATCH_SIZE = 1000  # for Mongo insert/update chunking


# ---------------- SQLite helpers ----------------
def get_sqlite_conn(db_path=DB_PATH):
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


def setup_sqlite_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            country TEXT
        )
    """)
    conn.commit()
    cur.close()


def sqlite_crud_from_df(df: pd.DataFrame, db_path=DB_PATH) -> Dict[str, float]:
    """
    Perform CRUD on SQLite using the provided DataFrame.
    Expects df to have columns: CustomerID, Country (case-insensitive).
    Returns timings for Create, Read, Update, Delete.
    """
    # Normalize column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    if "customerid" not in cols:
        raise ValueError("DataFrame must contain a 'CustomerID' column")
    # Country is optional; fill with empty string if missing
    country_col = cols.get("country", None)

    # Prepare clean df
    df = df.copy()
    df = df[df[cols["customerid"]].notna()]  # drop rows with missing CustomerID
    df[cols["customerid"]] = df[cols["customerid"]].astype(int)
    if country_col:
        df[country_col] = df[country_col].fillna("").astype(str)
        df = df[[cols["customerid"], country_col]]
        df.columns = ["CustomerID", "Country"]
    else:
        df["Country"] = ""
        df = df[[cols["customerid"], "Country"]]
        df.columns = ["CustomerID", "Country"]

    # dedupe
    df = df.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)

    conn = get_sqlite_conn(db_path)
    setup_sqlite_table(conn)
    cur = conn.cursor()
    timings: Dict[str, float] = {}

    # Convert to list of tuples (id, country)
    insert_rows = [(int(r["CustomerID"]), r["Country"]) for _, r in df.iterrows()]
    num = len(insert_rows)

    # CREATE - batch insert using INSERT OR REPLACE to avoid PK conflict
    start = time.time()
    if insert_rows:
        cur.executemany("INSERT OR REPLACE INTO customers (customer_id, country) VALUES (?, ?)", insert_rows)
        conn.commit()
    timings["SQL Create"] = time.time() - start

    # READ - per-row selects (simulate many small reads)
    start = time.time()
    for cid, _ in insert_rows:
        cur.execute("SELECT customer_id, country FROM customers WHERE customer_id = ?", (cid,))
        _ = cur.fetchone()
    timings["SQL Read"] = time.time() - start

    # UPDATE - batch executemany: set country to UpdatedCountry_{i}
    update_rows = [(f"UpdatedCountry_{i}", insert_rows[i][0]) for i in range(num)]
    start = time.time()
    if update_rows:
        cur.executemany("UPDATE customers SET country = ? WHERE customer_id = ?", update_rows)
        conn.commit()
    timings["SQL Update"] = time.time() - start

    # DELETE - batch delete by ids (use correct table/column names)
    id_tuples = [(insert_rows[i][0],) for i in range(num)]
    start = time.time()
    if id_tuples:
        cur.executemany("DELETE FROM customers WHERE customer_id = ?", id_tuples)
        conn.commit()
    timings["SQL Delete"] = time.time() - start

    cur.close()
    conn.close()
    return timings


# ---------------- MongoDB helpers ----------------
def get_mongo_client(uri=MONGO_URI, timeout_ms=2000) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def mongo_crud_from_df(df: pd.DataFrame, uri=MONGO_URI, db_name=MONGO_DB) -> Optional[Dict[str, float]]:
    """
    Perform CRUD operations on MongoDB collection 'customers_perf' using CustomerID values from df.
    Returns timings for Create, Read, Update, Delete or None if Mongo not reachable.
    """
    # Normalize and validate columns
    cols = {c.lower(): c for c in df.columns}
    if "customerid" not in cols:
        raise ValueError("DataFrame must contain a 'CustomerID' column")

    df = df.copy()
    df = df[df[cols["customerid"]].notna()]
    df[cols["customerid"]] = df[cols["customerid"]].astype(int)
    country_col = cols.get("country", None)
    if country_col:
        df[country_col] = df[country_col].fillna("").astype(str)
        df = df[[cols["customerid"], country_col]]
        df.columns = ["CustomerID", "Country"]
    else:
        df["Country"] = ""
        df = df[[cols["customerid"], "Country"]]
        df.columns = ["CustomerID", "Country"]

    # dedupe
    df = df.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)

    try:
        client = get_mongo_client(uri)
        client.admin.command("ping")
    except errors.PyMongoError as e:
        print("MongoDB not reachable:", e)
        return None

    db = client[db_name]
    coll = db["customers_perf"]
    timings: Dict[str, float] = {}

    # Clean collection
    coll.drop()

    # CREATE - build docs from df and insert in batches
    docs = [{"CustomerID": int(r["CustomerID"]), "Country": r["Country"], "Transactions": []} for _, r in df.iterrows()]
    start = time.time()
    if docs:
        if len(docs) <= BATCH_SIZE:
            coll.insert_many(docs)
        else:
            for i in range(0, len(docs), BATCH_SIZE):
                coll.insert_many(docs[i : i + BATCH_SIZE])
    timings["Mongo Create"] = time.time() - start

    # READ - per-record find_one using CustomerID values from the DataFrame
    start = time.time()
    for cid in df["CustomerID"].tolist():
        coll.find_one({"CustomerID": int(cid)})
    timings["Mongo Read"] = time.time() - start

    # UPDATE - bulk UpdateOne operations based on DataFrame order (set Country -> UpdatedCountry_i)
    bulk_ops = [UpdateOne({"CustomerID": int(r["CustomerID"])}, {"$set": {"Country": f"UpdatedCountry_{i}"}}) for i, r in enumerate(df.to_dict(orient="records"))]
    start = time.time()
    for i in range(0, len(bulk_ops), BATCH_SIZE):
        chunk = bulk_ops[i : i + BATCH_SIZE]
        if chunk:
            coll.bulk_write(chunk, ordered=False)
    timings["Mongo Update"] = time.time() - start

    # DELETE - delete_many by matching the CustomerID set inserted (use $in in chunks if needed)
    ids = [int(cid) for cid in df["CustomerID"].tolist()]
    start = time.time()
    if ids:
        # if very large list chunk deletes to avoid giant query
        for i in range(0, len(ids), BATCH_SIZE):
            coll.delete_many({"CustomerID": {"$in": ids[i : i + BATCH_SIZE]}})
    timings["Mongo Delete"] = time.time() - start

    client.close()
    return timings


# ---------------- Runner that uses given df ----------------
def run_crud_with_df(df: pd.DataFrame, db_path=DB_PATH, mongo_uri=MONGO_URI, mongo_db=MONGO_DB):
    """
    Run both SQLite and Mongo CRUD using the provided DataFrame.
    Prints timings and a comparison table.
    """
    # Optionally cap to BATCH_SIZE or user-provided cap; here we use full df
    # If you want to limit: df = df.head(n)

    print(f"Running CRUD with DataFrame of {len(df)} rows\nSQLite DB file: {db_path}\nMongo URI: {mongo_uri}\nMongo DB: {mongo_db}\n")

    print("Running SQLite CRUD (file-backed)...")
    sql_times = sqlite_crud_from_df(df, db_path)
    for k, v in sql_times.items():
        print(f"{k}: {v:.6f} s")

    print("\nRunning Mongo CRUD (if reachable)...")
    mongo_times = mongo_crud_from_df(df, mongo_uri, mongo_db)
    if mongo_times is None:
        print("MongoDB tests skipped (MongoDB not reachable).")
    else:
        for k, v in mongo_times.items():
            print(f"{k}: {v:.6f} s")

    # Print comparison table
    def print_comparison(sql_timings: Dict[str, float], mongo_timings: Optional[Dict[str, float]]):
        ops = ["Create", "Read", "Update", "Delete"]
        header = f"{'Operation':<10} | {'SQLite (s)':>12} | {'MongoDB (s)':>12} | {'Mongo/SQLite':>12}"
        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header))
        total_sql = 0.0
        total_mongo = 0.0
        for op in ops:
            k_sql = f"SQL {op}"
            k_m = f"Mongo {op}"
            sql_t = sql_timings.get(k_sql, float("nan"))
            mongo_t = mongo_timings.get(k_m) if mongo_timings else None
            ratio = (mongo_t / sql_t) if (mongo_t is not None and sql_t and sql_t > 0) else None
            total_sql += sql_t
            if mongo_t:
                total_mongo += mongo_t
            sql_s = f"{sql_t:.6f}"
            mongo_s = f"{mongo_t:.6f}" if mongo_t is not None else "N/A"
            ratio_s = f"{ratio:.2f}" if ratio is not None else "N/A"
            print(f"{op:<10} | {sql_s:>12} | {mongo_s:>12} | {ratio_s:>12}")
        print("-" * len(header))
        total_ratio = (total_mongo / total_sql) if (mongo_timings and total_sql > 0) else None
        total_sql_s = f"{total_sql:.6f}"
        total_mongo_s = f"{total_mongo:.6f}" if mongo_timings else "N/A"
        total_ratio_s = f"{total_ratio:.2f}" if total_ratio is not None else "N/A"
        print(f"{'TOTAL':<10} | {total_sql_s:>12} | {total_mongo_s:>12} | {total_ratio_s:>12}")
        print("-" * len(header))

    print_comparison(sql_times, mongo_times)

    # Quick human-friendly summary
    if mongo_times:
        if sum(mongo_times.values()) < sum(sql_times.values()):
            print("\nSummary: MongoDB was faster overall for this workload.")
        else:
            print("\nSummary: SQLite was faster overall for this workload.")
    else:
        print("\nSummary: MongoDB tests skipped; only SQLite timings available.")


if __name__ == "__main__":
    sample = pd.read_excel('Online Retail.xlsx')
    run_crud_with_df(sample.head(1000))


# Running CRUD with DataFrame of 1000 rows
# SQLite DB file: onlineretail1.db
# Mongo URI: mongodb://localhost:27017/
# Mongo DB: onlineretail
#
# Running SQLite CRUD (file-backed)...
# SQL Create: 0.003001 s
# SQL Read: 0.001521 s
# SQL Update: 0.002527 s
# SQL Delete: 0.002999 s
#
# Running Mongo CRUD (if reachable)...
# Mongo Create: 0.009027 s
# Mongo Read: 0.113780 s
# Mongo Update: 0.009031 s
# Mongo Delete: 0.003009 s
#
# -------------------------------------------------------
# Operation  |   SQLite (s) |  MongoDB (s) | Mongo/SQLite
# -------------------------------------------------------
# Create     |     0.003001 |     0.009027 |         3.01
# Read       |     0.001521 |     0.113780 |        74.79
# Update     |     0.002527 |     0.009031 |         3.57
# Delete     |     0.002999 |     0.003009 |         1.00
# -------------------------------------------------------
# TOTAL      |     0.010048 |     0.134847 |        13.42
# -------------------------------------------------------
#
# Summary: SQLite was faster overall for this workload.
