from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd

# ---------------- MongoDB Config ----------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "onlineretail"
COLLECTION_NAME = "transactions"

# ---------------- Connect MongoDB ----------------
def get_mongo_connection():
    try:
        client = MongoClient(MONGO_URI, maxPoolSize=50, wTimeoutMS=2500)
        client.admin.command("ping")  # Test connection
        print("✅ Connected to MongoDB")
        return client
    except ConnectionFailure:
        print("❌ MongoDB connection failed")
        return None

# ---------------- Transaction-Centric Insert ----------------
def insert_transaction_centric(csv_path, limit=1000):
    df = pd.read_excel(csv_path)
    df = df.dropna(subset=["CustomerID"])
    df = df.head(limit)

    client = get_mongo_connection()
    if client is None:
        return
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    # Clear old data
    coll.delete_many({})

    # Group by InvoiceNo
    grouped = df.groupby("InvoiceNo")

    docs = []
    for invoice, rows in grouped:
        first_row = rows.iloc[0]
        doc = {
            "InvoiceNo": str(invoice),
            "InvoiceDate": str(first_row["InvoiceDate"]),
            "CustomerID": int(first_row["CustomerID"]),
            "Country": first_row["Country"],
            "Products": [
                {
                    "StockCode": str(r["StockCode"]),
                    "Description": str(r["Description"]),
                    "Quantity": int(r["Quantity"]),
                    "UnitPrice": float(r["UnitPrice"])
                }
                for _, r in rows.iterrows()
            ]
        }
        docs.append(doc)

    coll.insert_many(docs)
    print(f"✅ Inserted {len(docs)} transaction documents into MongoDB (Transaction-Centric).")
    client.close()

# ---------------- Run Part 2A ----------------
if __name__ == "__main__":
    insert_transaction_centric('Online Retail.xlsx', limit=1000)

