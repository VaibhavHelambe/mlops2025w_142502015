from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd

# ---------------- MongoDB Config ----------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "onlineretail"
COLLECTION_NAME = "customers"

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

# ---------------- Customer-Centric Insert ----------------
def insert_customer_centric(path, limit=1000):
    df = pd.read_excel(path)
    df = df.dropna(subset=["CustomerID"])
    df = df.head(limit)

    client = get_mongo_connection()
    if client is None:
        return
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    # Clear old data
    coll.delete_many({})

    # Group by CustomerID
    grouped = df.groupby("CustomerID")

    docs = []
    for cid, rows in grouped:
        first_row = rows.iloc[0]
        doc = {
            "CustomerID": int(cid),
            "Country": first_row["Country"],
            "Transactions": []
        }

        # Group that customer's transactions by InvoiceNo
        inv_groups = rows.groupby("InvoiceNo")
        for inv, inv_rows in inv_groups:
            tdoc = {
                "InvoiceNo": str(inv),
                "InvoiceDate": str(inv_rows.iloc[0]["InvoiceDate"]),
                "Products": [
                    {
                        "StockCode": str(r["StockCode"]),
                        "Description": str(r["Description"]),
                        "Quantity": int(r["Quantity"]),
                        "UnitPrice": float(r["UnitPrice"])
                    }
                    for _, r in inv_rows.iterrows()
                ]
            }
            doc["Transactions"].append(tdoc)

        docs.append(doc)

    coll.insert_many(docs)
    print(f"✅ Inserted {len(docs)} customer documents into MongoDB (Customer-Centric).")
    client.close()

# ---------------- Run Part 2B ----------------
if __name__ == "__main__":
    insert_customer_centric('Online Retail.xlsx', limit=1000)

