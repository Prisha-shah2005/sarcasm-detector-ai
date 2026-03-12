import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model", "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl"))

def predict_sarcasm(text: str):

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    probabilities = model.predict_proba(vec)[0]

    confidence = round(max(probabilities) * 100, 2)

    label = "Normal" if prediction == 1 else "Sarcasm"

    return {
        "prediction": label,
        "confidence": f"{confidence}%"
    }