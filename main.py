from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import pymysql
import os
import io
import csv

app = FastAPI(title="Machine Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306))
}

scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_model.pkl")

THRESHOLD = 0.30

feature_cols = [
    "Temperature_C",
    "Pressure_bar",
    "Vibration_Level",
    "Sound_dB",
    "Humidity_%",
    "Load_Percentage"
]


class SensorData(BaseModel):
    temperature_c: float
    pressure_bar: float
    vibration_level: float
    sound_db: float
    humidity_pct: float


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


@app.get("/")
def home():
    return {"message": "Machine Monitoring API is running"}

@app.get("/debug-db")
def debug_db():
    return {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD_SET": os.getenv("DB_PASSWORD") is not None
    }


@app.get("/init-db")
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_sensor_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        temperature_c FLOAT NOT NULL,
        pressure_bar FLOAT NOT NULL,
        vibration_level FLOAT NOT NULL,
        sound_db FLOAT NOT NULL,
        humidity_percent FLOAT NOT NULL,
        load_percentage FLOAT DEFAULT 50,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        raw_data_id INT,
        model_name VARCHAR(100),
        prediction INT NOT NULL,
        status VARCHAR(20),
        probability FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (raw_data_id) REFERENCES raw_sensor_data(id)
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "Tables created successfully"}


@app.get("/fix-db")
def fix_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    messages = []

    try:
        cursor.execute("""
        ALTER TABLE raw_sensor_data
        ADD COLUMN load_percentage FLOAT DEFAULT 50
        """)
        messages.append("load_percentage column added")
    except Exception as e:
        messages.append("load_percentage check: " + str(e))

    try:
        cursor.execute("""
        UPDATE raw_sensor_data
        SET load_percentage = 50
        WHERE load_percentage IS NULL
        """)
        messages.append("NULL load_percentage values updated to 50")
    except Exception as e:
        messages.append("update check: " + str(e))

    conn.commit()
    cursor.close()
    conn.close()

    return {"messages": messages}


@app.post("/predict")
def predict(data: SensorData):
    load_percentage = 50.0

    input_df = pd.DataFrame([{
        "Temperature_C": data.temperature_c,
        "Pressure_bar": data.pressure_bar,
        "Vibration_Level": data.vibration_level,
        "Sound_dB": data.sound_db,
        "Humidity_%": data.humidity_pct,
        "Load_Percentage": load_percentage
    }])

    input_scaled = scaler.transform(input_df[feature_cols])
    probability = model.predict_proba(input_scaled)[0][1]

    prediction = 1 if probability >= THRESHOLD else 0
    status = "FAULT" if prediction == 1 else "NORMAL"

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_raw = """
    INSERT INTO raw_sensor_data
    (temperature_c, pressure_bar, vibration_level, sound_db, humidity_percent, load_percentage)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    raw_values = (
        data.temperature_c,
        data.pressure_bar,
        data.vibration_level,
        data.sound_db,
        data.humidity_pct,
        load_percentage
    )

    cursor.execute(insert_raw, raw_values)
    raw_data_id = cursor.lastrowid

    insert_prediction = """
    INSERT INTO predictions
    (raw_data_id, model_name, prediction, status, probability)
    VALUES (%s, %s, %s, %s, %s)
    """

    prediction_values = (
        raw_data_id,
        "Logistic Regression",
        prediction,
        status,
        float(probability)
    )

    cursor.execute(insert_prediction, prediction_values)

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "id": raw_data_id,
        "model": "Logistic Regression",
        "threshold": THRESHOLD,
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "status": status,
        "temperature_c": data.temperature_c,
        "pressure_bar": data.pressure_bar,
        "vibration_level": data.vibration_level,
        "sound_db": data.sound_db,
        "humidity_percent": data.humidity_pct,
        "load_percentage": load_percentage
    }


@app.get("/latest")
def latest():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        r.id,
        r.temperature_c,
        r.pressure_bar,
        r.vibration_level,
        r.sound_db,
        r.humidity_percent,
        r.load_percentage,
        r.created_at,
        p.model_name,
        p.prediction,
        p.status,
        p.probability
    FROM raw_sensor_data r
    LEFT JOIN predictions p ON r.id = p.raw_data_id
    ORDER BY r.id DESC
    LIMIT 1
    """

    cursor.execute(query)
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result


@app.get("/history")
def history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        r.id,
        r.temperature_c,
        r.pressure_bar,
        r.vibration_level,
        r.sound_db,
        r.humidity_percent,
        r.load_percentage,
        r.created_at,
        p.model_name,
        p.prediction,
        p.status,
        p.probability
    FROM raw_sensor_data r
    LEFT JOIN predictions p ON r.id = p.raw_data_id
    WHERE p.status = 'FAULT'
    ORDER BY r.id DESC
    LIMIT 50
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


@app.get("/dataset")
def dataset():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT 
        r.temperature_c,
        r.pressure_bar,
        r.vibration_level,
        r.sound_db,
        r.humidity_percent,
        r.load_percentage,
        p.prediction AS target,
        p.status,
        p.probability,
        r.created_at
    FROM raw_sensor_data r
    LEFT JOIN predictions p ON r.id = p.raw_data_id
    ORDER BY r.id ASC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


@app.get("/dataset-csv")
def dataset_csv():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT 
        r.temperature_c,
        r.pressure_bar,
        r.vibration_level,
        r.sound_db,
        r.humidity_percent,
        r.load_percentage,
        p.prediction AS target,
        p.status,
        p.probability,
        r.created_at
    FROM raw_sensor_data r
    LEFT JOIN predictions p ON r.id = p.raw_data_id
    ORDER BY r.id ASC
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    output = io.StringIO()

    fieldnames = [
        "temperature_c",
        "pressure_bar",
        "vibration_level",
        "sound_db",
        "humidity_percent",
        "load_percentage",
        "target",
        "status",
        "probability",
        "created_at"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sensor_dataset.csv"}
    )
