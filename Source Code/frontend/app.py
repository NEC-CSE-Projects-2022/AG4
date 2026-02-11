from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LayerNormalization, Layer

# -------------------------------
# Define your custom HREFBlock layer
# -------------------------------
import tensorflow as tf
from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense, LayerNormalization, Layer

class HREFBlock(Layer):
    def __init__(self, units=256, **kwargs):
        super(HREFBlock, self).__init__(**kwargs)
        self.units = units

        # Internal layers
        self.dense1 = Dense(units, activation='relu')
        self.norm1 = LayerNormalization()

        self.dense2 = Dense(units, activation='relu')
        self.norm2 = LayerNormalization()

        self.projection = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.projection = Dense(self.units)
        else:
            self.projection = None
        super().build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.norm1(x)

        x = self.dense2(x)
        x = self.norm2(x)

        shortcut = self.projection(inputs) if self.projection else inputs

        return tf.keras.activations.relu(x + shortcut)

    def get_config(self):
        config = super(HREFBlock, self).get_config()
        config.update({
            "units": self.units
        })
        return config


# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# Load model with custom layer
model = load_model("final_model.h5", custom_objects={'HREFBlock': HREFBlock})

# -------------------------------
# STORE PREDICTION FUNCTION (PASTE HERE)
# -------------------------------
def store_prediction(inputs, prob, result):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions
    (age, sex, cp, trestbps, chol, fbs, restecg,
     thalach, exang, oldpeak, slope, ca, thal,
     probability, result)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (*inputs, prob, result))

    # ✅ keep only last 100 records
    cursor.execute("""
    DELETE FROM predictions
    WHERE id NOT IN (
        SELECT id FROM predictions
        ORDER BY id DESC
        LIMIT 100
    )
    """)

    conn.commit()
    conn.close()



@app.route("/")
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')

def generate_graph(prob):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import uuid

    disease = round(prob * 100, 2)
    no_disease = round((1 - prob) * 100, 2)

    labels = ["No Disease", "Heart Disease"]
    values = [no_disease, disease]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)

    plt.ylim(0, 100)
    plt.ylabel("Probability (%)")
    plt.title(f"Prediction Probability (Disease = {prob:.2f})")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value}%",
            ha="center"
        )

    # ✅ Correct absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(base_dir, "static", "graphs")

    # ✅ Create folder automatically if not exists
    os.makedirs(graphs_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(graphs_dir, filename)

    plt.savefig(filepath)
    plt.close()

    return filename

@app.route('/dataset')
def dataset_page():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    return render_template("dataset.html", rows=rows)

@app.route("/delete_predictions", methods=["POST"])
def delete_predictions():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return "Deleted Successfully"


@app.route("/bulk_upload", methods=["GET", "POST"])
def bulk_upload():
    REQUIRED_COLUMNS = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    if request.method == "POST":
        file = request.files.get("file")

        # 1) Validation: file selected
        if not file or file.filename == "":
            return render_template("bulk_upload.html", error="❌ Please select a CSV file")

        # 2) Validation: only CSV allowed
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".csv"):
            return render_template("bulk_upload.html", error="❌ Only CSV files are allowed")

        try:
            # 3) Read CSV
            df = pd.read_csv(file)

            # 4) Validation: empty CSV
            if df.empty:
                return render_template("bulk_upload.html", error="❌ Uploaded CSV is empty")

            # 5) Validation: required columns
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                return render_template(
                    "bulk_upload.html",
                    error=f"❌ Missing columns: {', '.join(missing)}"
                )

            # 6) Remove empty rows
            df = df.dropna(subset=REQUIRED_COLUMNS)

            # 7) Convert all to numeric
            for col in REQUIRED_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 8) Remove invalid numeric rows
            df = df.dropna(subset=REQUIRED_COLUMNS)

            # 9) Validation: if no valid rows
            if df.empty:
                return render_template("bulk_upload.html", error="❌ No valid rows found")

            # 10) Insert predicted results into DB
            conn = sqlite3.connect("predictions.db",timeout=30)
            cursor = conn.cursor()

            inserted = 0

            for _, row in df.iterrows():
                # Create input for model
                x = np.array([
                    row["age"], row["sex"], row["cp"], row["trestbps"], row["chol"],
                    row["fbs"], row["restecg"], row["thalach"], row["exang"],
                    row["oldpeak"], row["slope"], row["ca"], row["thal"]
                ], dtype=np.float32).reshape(1, -1)

                # Predict
                prediction = model.predict(x, verbose=0)
                prob = float(prediction[0][0])

                # Result
                result = "Heart Disease" if prob >= 0.46 else "No Heart Disease"

                # Store
                cursor.execute("""
                INSERT INTO predictions
                (age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope, ca, thal,
                 probability, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    float(row["age"]),
                    float(row["sex"]),
                    float(row["cp"]),
                    float(row["trestbps"]),
                    float(row["chol"]),
                    float(row["fbs"]),
                    float(row["restecg"]),
                    float(row["thalach"]),
                    float(row["exang"]),
                    float(row["oldpeak"]),
                    float(row["slope"]),
                    float(row["ca"]),
                    float(row["thal"]),
                    prob,
                    result
                ))

                inserted += 1

            conn.commit()
            conn.close()
            

            return render_template(
                "bulk_upload.html",
                success=f"✅ Bulk Prediction Completed! Rows predicted & stored: {inserted}",
               
             )

        except Exception as e:
            return render_template("bulk_upload.html", error=f"❌ Error: {str(e)}")

    return render_template("bulk_upload.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None or 'input' not in data:
            return jsonify({"error": "Please send JSON with 'input' key"}), 400

        x = np.array(data['input'])
        if x.ndim == 1:
            x = x.reshape(1, -1)

        prediction = model.predict(x)
        prob = float(prediction[0][0])
        if prob < 0.30:
           risk = "Low Risk"
        elif prob < 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        result = " Heart Disease" if prob >= 0.46 else " No Heart Disease"
        store_prediction(data['input'], prob, result)
        graph_file = generate_graph(prob)
        



        return render_template(
            "result.html",
            probability=round(prob, 4),
            risk=risk,
            result=result,
            graph_file=graph_file
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
