import sqlite3

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak REAL,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    probability REAL,
    result TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")
