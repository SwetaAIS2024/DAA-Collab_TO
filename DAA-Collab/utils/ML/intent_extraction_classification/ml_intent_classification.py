import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib
import os
from typing import List

DATA_PATH = "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv"
MODEL_DIR = "DAA-Collab/utils/ML/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_intent_classifier():
    """Train and save the ML intent classifier"""
    df = pd.read_csv(DATA_PATH)
    df["intents"] = df["intents"].apply(literal_eval)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["intents"])

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["prompt"])

    classifier = OneVsRestClassifier(LinearSVC(class_weight="balanced"))
    classifier.fit(X, Y)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "intent_vectorizer.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "intent_classifier.joblib"))
    joblib.dump(mlb, os.path.join(MODEL_DIR, "intent_mlb.joblib"))

    print("✅ Intent classifier trained and saved.")


def load_intent_classifier():
    """Load the trained ML classifier"""
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "intent_vectorizer.joblib"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "intent_classifier.joblib"))
    mlb = joblib.load(os.path.join(MODEL_DIR, "intent_mlb.joblib"))
    return vectorizer, classifier, mlb


def predict_intents_ml(prompt: str, threshold: float = 0.0) -> List[str]:
    """
    Predict intents using the trained ML model.
    Returns a list of predicted intent labels.
    """
    try:
        vectorizer, classifier, mlb = load_intent_classifier()
        X = vectorizer.transform([prompt])
        # y_pred = classifier.predict(X)

        # using the decision function for better thresholding
        decision_scores = classifier.decision_function(X)
        y_pred = (decision_scores > threshold).astype(int)

        intents = mlb.inverse_transform(y_pred)
        return list(intents[0]) if intents and len(intents[0]) > 0 else []
    except FileNotFoundError:
        print("⚠️ ML models not found. Run train_intent_classifier() first.")
        return []
    except Exception as e:
        print(f"⚠️ ML prediction error: {e}")
        return []


if __name__ == "__main__":
    train_intent_classifier()