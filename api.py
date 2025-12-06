from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# ========================================================
# LOAD MODEL
# ========================================================
model_bundle = pickle.load(open("PHEmail-Model.pkl", "rb"))
model = model_bundle["model"]
tfidf = model_bundle["tfidf"]

print("🔥 Model Loaded Successfully!")

# ========================================================
# FASTAPI APP
# ========================================================
app = FastAPI()

# ========================================================
# ENABLE CORS FOR ALL ORIGINS
# ========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # allow all websites
    allow_credentials=True,
    allow_methods=["*"],           # allow POST, GET, OPTIONS...
    allow_headers=["*"],
)

# ========================================================
# INPUT SCHEMA
# ========================================================
class EmailInput(BaseModel):
    sender: str
    subject: str
    body: str

# ========================================================
# ROOT ENDPOINT
# ========================================================
@app.get("/")
def home():
    return {"message": "PHEmail API is running!"}

# ========================================================
# PREDICT ENDPOINT
# ========================================================
@app.post("/predict")
def predict(email: EmailInput):
    full_text = email.subject + " " + email.body
    X = tfidf.transform([full_text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": "phishing" if pred == 1 else "legitimate",
        "phishing_probability": float(prob)
    }
