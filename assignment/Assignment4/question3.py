"""
CRUD comparison: file-backed SQLite vs MongoDB (if available)

Usage:
    python compare_crud.py

Requirements:
    - Python 3.8+
    - pymongo (optional, only needed if you want to test Mongo): pip install pymongo
"""
import sqlite3
import time
from typing import Dict, Optional
from pymongo import MongoClient, errors
from pymongo import UpdateOne

# Config
DB_PATH = "onlineretail1.db"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "online_retail_perf_test"
NUM_RECORDS = 100
BASE_ID = 200000  # test CustomerID starting number
BATCH_SIZE = 1000  # for Mongo insert chunking (not critical for 100 rows)


# ---------------- SQLite (file-backed) ----------------
def get_sqlite_conn(db_path=DB_PATH):
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


def setup_sqlite_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Customers (
            CustomerID INTEGER PRIMARY KEY,
            Country TEXT
        )
    """)
    conn.commit()
    cur.close()


def sqlite_crud(num=NUM_RECORDS, base_id=BASE_ID, db_path=DB_PATH) -> Dict[str, float]:
    conn = get_sqlite_conn(db_path)
    setup_sqlite_table(conn)
    cur = conn.cursor()

    timings: Dict[str, float] = {}

    # CREATE - batch insert
    rows = [(base_id + i, f"Country_{i}") for i in range(num)]
    start = time.time()
    cur.executemany("INSERT OR REPLACE INTO Customers (CustomerID, Country) VALUES (?, ?)", rows)
    conn.commit()
    timings["SQL Create"] = time.time() - start

    # READ - per-row selects
    start = time.time()
    for i in range(num):
        cid = base_id + i
        cur.execute("SELECT CustomerID, Country FROM Customers WHERE CustomerID = ?", (cid,))
        _ = cur.fetchone()
    timings["SQL Read"] = time.time() - start

    # UPDATE - batch executemany
    update_rows = [(f"UpdatedCountry_{i}", base_id + i) for i in range(num)]
    start = time.time()
    cur.executemany("UPDATE Customers SET Country = ? WHERE CustomerID = ?", update_rows)
    conn.commit()
    timings["SQL Update"] = time.time() - start

    # DELETE - batch delete by ids
    id_tuples = [(base_id + i,) for i in range(num)]
    start = time.time()
    cur.executemany("DELETE FROM Customers WHERE CustomerID = ?", id_tuples)
    conn.commit()
    timings["SQL Delete"] = time.time() - start

    cur.close()
    conn.close()
    return timings


# ---------------- MongoDB ----------------
def get_mongo_client(uri=MONGO_URI, timeout_ms=2000) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def mongo_crud(num=NUM_RECORDS, base_id=BASE_ID, uri=MONGO_URI, db_name=MONGO_DB) -> Optional[Dict[str, float]]:
    try:
        client = get_mongo_client(uri)
        # quick ping to fail fast if not reachable
        client.admin.command("ping")
    except errors.PyMongoError as e:
        print("MongoDB not reachable:", e)
        return None

    db = client[db_name]
    coll = db["customers_perf"]
    timings: Dict[str, float] = {}

    # ensure clean collection
    coll.drop()

    # CREATE - insert_many in chunks
    docs = [{"CustomerID": base_id + i, "Country": f"Country_{i}", "Transactions": []} for i in range(num)]
    start = time.time()
    if len(docs) <= BATCH_SIZE:
        coll.insert_many(docs)
    else:
        for i in range(0, len(docs), BATCH_SIZE):
            coll.insert_many(docs[i:i + BATCH_SIZE])
    timings["Mongo Create"] = time.time() - start

    # READ - per-record find_one
    start = time.time()
    for i in range(num):
        cid = base_id + i
        coll.find_one({"CustomerID": cid})
    timings["Mongo Read"] = time.time() - start

    # UPDATE - use bulk writes of UpdateOne to mirror per-row updates but in a single bulk call
    bulk_ops = [UpdateOne({"CustomerID": base_id + i}, {"$set": {"Country": f"UpdatedCountry_{i}"}}) for i in range(num)]
    start = time.time()
    # perform bulk in chunks to avoid massive single bulk for very large N
    for i in range(0, len(bulk_ops), BATCH_SIZE):
        coll.bulk_write(bulk_ops[i:i + BATCH_SIZE], ordered=False)
    timings["Mongo Update"] = time.time() - start

    # DELETE - delete_many using list of ids
    start = time.time()
    coll.delete_many({"CustomerID": {"$gte": base_id, "$lt": base_id + num}})
    timings["Mongo Delete"] = time.time() - start

    # cleanup
    client.close()
    return timings


# ---------------- Comparator / Runner ----------------
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


if __name__ == "__main__":
    print(f"Running CRUD comparison with NUM_RECORDS = {NUM_RECORDS}\nSQLite DB file: {DB_PATH}\nMongo URI: {MONGO_URI}\n")

    print("Running SQLite CRUD (file-backed)...")
    sql_times = sqlite_crud(NUM_RECORDS, BASE_ID, DB_PATH)
    for k, v in sql_times.items():
        print(f"{k}: {v:.6f} s")

    print("\nRunning Mongo CRUD (if reachable)...")
    mongo_times = mongo_crud(NUM_RECORDS, BASE_ID, MONGO_URI, MONGO_DB)
    if mongo_times is None:
        print("MongoDB tests skipped (MongoDB not reachable).")
    else:
        for k, v in mongo_times.items():
            print(f"{k}: {v:.6f} s")

    # Print comparison table
    print_comparison(sql_times, mongo_times)

    # Quick human-friendly summary
    if mongo_times:
        if sum(mongo_times.values()) < sum(sql_times.values()):
            print("\nSummary: MongoDB was faster overall for this workload.")
        else:
            print("\nSummary: SQLite was faster overall for this workload.")
    else:
        print("\nSummary: MongoDB tests skipped; only SQLite timings available.")


#Running CRUD comparison with NUM_RECORDS = 100
# SQLite DB file: onlineretail1.db
# Mongo URI: mongodb://localhost:27017/
#
# Running SQLite CRUD (file-backed)...
# SQL Create: 0.003007 s
# SQL Read: 0.002999 s
# SQL Update: 0.003097 s
# SQL Delete: 0.002005 s
#
# Running Mongo CRUD (if reachable)...
# Mongo Create: 0.006811 s
# Mongo Read: 0.035153 s
# Mongo Update: 0.012173 s
# Mongo Delete: 0.001509 s
#
# -------------------------------------------------------
# Operation  |   SQLite (s) |  MongoDB (s) | Mongo/SQLite
# -------------------------------------------------------
# Create     |     0.003007 |     0.006811 |         2.27
# Read       |     0.002999 |     0.035153 |        11.72
# Update     |     0.003097 |     0.012173 |         3.93
# Delete     |     0.002005 |     0.001509 |         0.75
# -------------------------------------------------------
# TOTAL      |     0.011108 |     0.055646 |         5.01
# -------------------------------------------------------
#
# Summary: SQLite was faster overall for this workload.
