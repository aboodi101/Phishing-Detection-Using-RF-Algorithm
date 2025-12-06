from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# =====================================================
# LOAD MODEL + TF-IDF
# =====================================================
try:
    bundle = pickle.load(open("PHEmail-Model.pkl", "rb"))
    model = bundle["model"]
    tfidf = bundle["tfidf"]
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ ERROR loading model:", e)
    raise e

# =====================================================
# FASTAPI APP INITIALIZATION
# =====================================================
app = FastAPI(
    title="PHEmail Phishing Detection API",
    description="Machine-learning phishing detector using Random Forest + TF-IDF",
    version="1.0.0"
)

# =====================================================
# REQUEST MODEL
# =====================================================
class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

# =====================================================
# PREDICTION ROUTE
# =====================================================
@app.post("/predict")
def predict(email: EmailRequest):

    # Combine text
    text = email.subject + " " + email.body
    X = tfidf.transform([text])

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    result = {
        "prediction": "phishing" if pred == 1 else "legitimate",
        "confidence": float(probs[1])  # probability of phishing
    }

    return result

# =====================================================
# ROOT CHECK
# =====================================================
@app.get("/")
def home():
    return {"message": "PHEmail API is running successfully!"}
