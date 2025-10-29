from pymongo import MongoClient
import random

client = MongoClient('mongodb://localhost:27017/')
db = client['test_db']
col = db['users']

names = ["Alice", "Bob", "Charlie", "David", "Eve"]
cities = ["Mumbai", "Delhi", "Pune", "Bangalore", "Chennai"]

data = [
    {
        "name": random.choice(names),
        "email": f"user{i}@example.com",
        "age": random.randint(18, 60),
        "city": random.choice(cities)
    }
    for i in range(10)
]

col.insert_many(data)

for doc in col.find().limit(5):
    print(doc)
