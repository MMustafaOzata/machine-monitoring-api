# Import FastAPI tools
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import data validation model
from pydantic import BaseModel

# Import data and ML libraries
import pandas as pd
import joblib

# Import MySQL connection library
import pymysql

# Import system and file libraries
import os
import io
import csv


# Create FastAPI application
app = FastAPI(title="Machine Monitoring API")


# Allow requests from frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database connection information
# These values come from Render environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306))
}


# Load trained scaler and machine learning model
scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_model.pkl")


# Threshold value for fault prediction
THRESHOLD = 0.5


# Features used by the trained machine learning model
feature_cols = [
    "Temperature_C",
    "Pressure_bar",
    "Vibration_Level",
    "Sound_dB",
    "Humidity_%",
    "Load_Percentage"
]


# Input data format for sensor values
class SensorData(BaseModel):
    temperature_c: float
    pressure_bar: float
    vibration_level: float
    sound_db: float
    humidity_pct: float


# Create database connection
def get_db_connection():
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        port=DB_CONFIG["port"],
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl": {}}
    )


# Simple home endpoint
@app.get("/")
def home():
    return {"message": "Machine Monitoring API is running"}


# Create database tables if they do not exist
@app.get("/init-db")
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # This table stores raw sensor values
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

    # This table stores prediction results
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


# Clear all records from database
# This endpoint is used only for testing and demo preparation
@app.get("/clear-db")
def clear_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")
    cursor.execute("DELETE FROM raw_sensor_data")

    cursor.execute("ALTER TABLE predictions AUTO_INCREMENT = 1")
    cursor.execute("ALTER TABLE raw_sensor_data AUTO_INCREMENT = 1")

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "message": "Database reset successfully",
        "predictions_deleted": True,
        "sensor_data_deleted": True,
        "id_reset": True
    }


# Main prediction endpoint
@app.post("/predict")
def predict(data: SensorData):

    # Convert prototype sensor values to industrial working ranges
    temperature = data.temperature_c + 35
    pressure = data.pressure_bar
    vibration = data.vibration_level * 15
    sound = data.sound_db + 25
    humidity = data.humidity_pct

    # Load Percentage is not measured by physical sensor
    # So we use a fixed value for model compatibility
    load_percentage = 50.0

    # Prepare input data for the ML model
    input_df = pd.DataFrame([{
        "Temperature_C": temperature,
        "Pressure_bar": pressure,
        "Vibration_Level": vibration,
        "Sound_dB": sound,
        "Humidity_%": humidity,
        "Load_Percentage": load_percentage
    }])

    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_df[feature_cols])

    # Get fault probability from Logistic Regression model
    probability = model.predict_proba(input_scaled)[0][1]

    # Convert probability into class using threshold
    model_prediction = 1 if probability >= THRESHOLD else 0

    # Extra safety rule for very risky sensor values
    rule_fault = (
        temperature >= 90 or
        pressure >= 220 or
        vibration >= 8 or
        sound >= 90 or
        humidity >= 85
    )

    # Final decision
    if rule_fault:
        prediction = 1
        status = "FAULT"
        decision_source = "Rule-Based Safety Check"
    else:
        prediction = model_prediction
        status = "FAULT" if prediction == 1 else "NORMAL"
        decision_source = "Logistic Regression Model"

    # Save sensor values and prediction result to MySQL
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert raw sensor data
    cursor.execute("""
    INSERT INTO raw_sensor_data
    (temperature_c, pressure_bar, vibration_level, sound_db, humidity_percent, load_percentage)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        data.temperature_c,
        data.pressure_bar,
        data.vibration_level,
        data.sound_db,
        data.humidity_pct,
        load_percentage
    ))

    raw_data_id = cursor.lastrowid

    # Insert prediction result
    cursor.execute("""
    INSERT INTO predictions
    (raw_data_id, model_name, prediction, status, probability)
    VALUES (%s, %s, %s, %s, %s)
    """, (
        raw_data_id,
        "Logistic Regression + Rule Check",
        prediction,
        status,
        float(probability)
    ))

    conn.commit()
    cursor.close()
    conn.close()

    # Return prediction result to dashboard
    return {
        "id": raw_data_id,
        "model": "Logistic Regression + Rule Check",
        "decision_source": decision_source,
        "threshold": THRESHOLD,
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "status": status,
        "temperature_c": data.temperature_c,
        "pressure_bar": data.pressure_bar,
        "vibration_level": data.vibration_level,
        "sound_db": data.sound_db,
        "humidity_percent": data.humidity_pct,
        "load_percentage": load_percentage,
        "model_input_values": {
            "Temperature_C": temperature,
            "Pressure_bar": pressure,
            "Vibration_Level": vibration,
            "Sound_dB": sound,
            "Humidity_%": humidity,
            "Load_Percentage": load_percentage
        }
    }


# Get the latest sensor data and prediction
@app.get("/latest")
def latest():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
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
    """)

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result


# Get last 50 fault records
@app.get("/history")
def history():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
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
    """)

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


# Get all stored data as JSON
@app.get("/dataset")
def dataset():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
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
    """)

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


# Export stored data as CSV file
@app.get("/dataset-csv")
def dataset_csv():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
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
    """)

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
