from pymongo import MongoClient
import pandas as pd

# ---------------- MongoDB Atlas Config ----------------
MONGO_URI = "mongodb+srv://.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" # not my real username and pass i will import it from .env later
DB_NAME = "online_retail_atlas"
COLLECTION_NAME = "customers"

# ---------------- Connect to Atlas ----------------
def get_atlas_connection():
    client = MongoClient(MONGO_URI, maxPoolSize=50, wTimeoutMS=2500)
    client.admin.command("ping")  # Test connection
    print("✅ Connected to MongoDB Atlas Cluster")
    return client

# ---------------- Insert Customer-Centric Data ----------------
def insert_atlas_customer_centric(path, limit=1000):
    df = pd.read_excel(path)
    df = df.dropna(subset=["CustomerID"])
    df = df.head(limit)

    client = get_atlas_connection()
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
    print(f"✅ Inserted {len(docs)} documents into Atlas (Customer-Centric).")
    client.close()

# ---------------- Run Part 4 ----------------
if __name__ == "__main__":
    insert_atlas_customer_centric('Online Retail.xlsx', limit=1000)
