import sqlite3

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age REAL,
    sex REAL,
    cp REAL,
    trestbps REAL,
    chol REAL,
    fbs REAL,
    restecg REAL,
    thalach REAL,
    exang REAL,
    oldpeak REAL,
    slope REAL,
    ca REAL,
    thal REAL,
    probability REAL,
    result TEXT
)
""")

conn.commit()
conn.close()

print("Table created successfully!")
