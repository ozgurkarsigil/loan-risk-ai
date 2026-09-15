from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

THRESHOLD = 0.1693532262785989
pipeline = joblib.load("modelMLP.pkl")


class LoanRequest(BaseModel):
    Income: int
    LoanTerm: int
    LoanAmount: int
    InterestRate: float
    CreditScore: int
    MonthsEmployed: int
    DTIRatio: float


class PredictionResponse(BaseModel):
    default_probability: float
    predicted_class: int
    risk_level: str


logging.basicConfig(level=logging.INFO)


@app.get("/")
def home():
    return {"message": "API çalışıyor"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: LoanRequest):

    logging.info("Yeni prediction isteği geldi.")

    try:
        customer_df = pd.DataFrame([{
            "Income": customer.Income,
            "LoanTerm": customer.LoanTerm,
            "LoanAmount": customer.LoanAmount,
            "InterestRate": customer.InterestRate,
            "CreditScore": customer.CreditScore,
            "MonthsEmployed": customer.MonthsEmployed,
            "DTIRatio": customer.DTIRatio
        }])

        probability = pipeline.predict_proba(customer_df)[0][1]
        prediction = int(probability >= THRESHOLD)
        risk_level="high" if prediction==1 else "low"
        save_prediction(customer,probability,prediction,risk_level)

        return PredictionResponse(
            default_probability=round(float(probability), 4),
            predicted_class=prediction,
            risk_level="high" if prediction == 1 else "low"
        )

    except Exception as e:
        logging.exception("Tahmin sırasında hata oluştu.")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
def save_prediction(customer, probability, prediction, risk_level):
    print("SAVE FUNCTION ÇAĞRILDI")
    conn = get_db_connection()
    cursor = conn.cursor()
    print("DB:",conn.database)

    query = """
        INSERT INTO predictions (
            income,
            loan_term,
            loan_amount,
            interest_rate,
            credit_score,
            months_employed,
            dti_ratio,
            default_probability,
            predicted_class,
            risk_level
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        customer.Income,
        customer.LoanTerm,
        customer.LoanAmount,
        customer.InterestRate,
        customer.CreditScore,
        customer.MonthsEmployed,
        customer.DTIRatio,
        float(probability),
        prediction,
        risk_level
    )
    print("VALUES:", values)

    cursor.execute(query, values)
    conn.commit()
    print("INSERTED ID:",cursor.lastrowid)
    cursor.close()
    conn.close()
