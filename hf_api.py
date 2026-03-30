"""
SentimentAI — FastAPI Backend for Hugging Face Spaces
======================================================
Trains on Twitter CSV dataset for high accuracy (same as local app.py).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentAI API",
    description="Sentiment analysis using Logistic Regression & Naive Bayes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_WORDS = 3
THRESHOLD = 55.0

# ── Models ────────────────────────────────────────────────────────────────────
MODEL = None
NB_MODEL = None
MODEL_METRICS = {}

# ── Text Cleaning ─────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "doesn't": "does not", "didn't": "did not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "he's": "he is", "she's": "she is",
}


def expand_contractions(text: str) -> str:
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    return text


def handle_negation(text: str) -> str:
    negations = {
        "not", "no", "never", "nothing", "nobody", "neither", "nor",
        "barely", "hardly", "scarcely", "cannot", "without"
    }
    words = text.split()
    result = []
    negate = False
    for word in words:
        cw = re.sub(r"[^a-z]", "", word)
        if cw in negations:
            negate = True
            result.append(word)
        elif negate and cw.isalpha():
            result.append(f"NOT_{word}")
            negate = False
        else:
            result.append(word)
            if word in [".", ",", "!", "?"]:
                negate = False
    return " ".join(result)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = expand_contractions(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = handle_negation(text)
    return text


# ── Load Dataset ──────────────────────────────────────────────────────────────
def load_dataset():
    col_names = ["id", "entity", "label", "text"]
    dfs = []
    for path in ["twitter_training.csv", "twitter_validation.csv"]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, header=None, names=col_names,
                                   encoding="utf-8", on_bad_lines="skip"))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 3]
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 3]

    SAMPLES = 8000
    parts = []
    for lbl in ["positive", "negative", "neutral"]:
        part = df[df["label"] == lbl]
        parts.append(part.sample(min(SAMPLES, len(part)), random_state=42))

    df_final = shuffle(pd.concat(parts, ignore_index=True), random_state=42)
    return df_final.reset_index(drop=True)


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    global MODEL, NB_MODEL, MODEL_METRICS

    print("Loading dataset...")
    df = load_dataset()

    if df is not None:
        print(f"✅ Twitter CSV loaded! {len(df)} samples")
        texts  = df["clean_text"].tolist()
        labels = df["label"].tolist()
        source = "twitter_csv"
    else:
        print("⚠️ CSV not found — using fallback data")
        # Fallback small dataset
        RAW_DATA = [
            ("I absolutely love this product it is amazing", "positive"),
            ("This is the best purchase I have ever made", "positive"),
            ("Amazing quality and super fast delivery", "positive"),
            ("The customer service was fantastic and helpful", "positive"),
            ("Great value for money exactly what I needed", "positive"),
            ("This is absolute garbage total waste of money", "negative"),
            ("Terrible quality broke after two days of use", "negative"),
            ("Worst purchase I have ever made do not buy", "negative"),
            ("Completely useless nothing works as advertised", "negative"),
            ("I am so frustrated customer support is terrible", "negative"),
            ("The product arrived on time and is as described", "neutral"),
            ("It does what it says nothing more nothing less", "neutral"),
            ("Average quality for the price works okay", "neutral"),
            ("Decent product has some good points and bad", "neutral"),
            ("It is fine not amazing not terrible just average", "neutral"),
        ]
        texts  = [clean_text(t) for t, _ in RAW_DATA]
        labels = [l for _, l in RAW_DATA]
        source = "fallback"

    tfidf_params = dict(ngram_range=(1, 2), max_features=50000,
                        sublinear_tf=True, min_df=2, strip_accents="unicode")

    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                     solver="lbfgs", random_state=42))
    ])
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   MultinomialNB(alpha=0.3))
    ])

    lr_pipe.fit(texts, labels)
    nb_pipe.fit(texts, labels)

    cv = min(5, len(set(labels)))
    lr_cv = cross_val_score(lr_pipe, texts, labels, cv=cv).mean()
    nb_cv = cross_val_score(nb_pipe, texts, labels, cv=cv).mean()

    MODEL    = lr_pipe
    NB_MODEL = nb_pipe
    MODEL_METRICS = {
        "logistic_regression": {"cv_accuracy": round(float(lr_cv), 4)},
        "naive_bayes":         {"cv_accuracy": round(float(nb_cv), 4)},
        "dataset_size":        len(texts),
        "source":              source,
    }
    print(f"✅ Models trained! LR CV: {lr_cv:.3f} | NB CV: {nb_cv:.3f} | Source: {source}")


train()


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "SentimentAI API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/analyze", "/batch", "/metrics", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/metrics")
def metrics():
    return MODEL_METRICS


@app.post("/analyze")
def analyze(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(body.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")

    cleaned = clean_text(body.text)

    if len(cleaned.split()) < MIN_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Text too short — please enter at least {MIN_WORDS} words."
        )

    results = {}
    for key, pipe, name in [("lr", MODEL, "Logistic Regression"),
                              ("nb", NB_MODEL, "Naive Bayes")]:
        pred  = pipe.predict([cleaned])[0]
        probs = pipe.predict_proba([cleaned])[0]
        classes = list(pipe.classes_)
        prob_dict = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}
        results[key] = {
            "model_name":    name,
            "prediction":    pred,
            "confidence":    round(float(max(probs)) * 100, 1),
            "probabilities": prob_dict,
        }

    max_conf = max(results["lr"]["confidence"], results["nb"]["confidence"])

    return {
        "text":       body.text,
        "word_count": len(body.text.split()),
        "char_count": len(body.text),
        "uncertain":  max_conf < THRESHOLD,
        "results":    results,
    }


@app.post("/batch")
def batch(body: BatchInput):
    if not body.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if len(body.texts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 texts per batch")

    output = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for text in body.texts:
        cleaned = clean_text(text)
        if len(cleaned.split()) < MIN_WORDS:
            output.append({
                "text":          text[:80] + ("..." if len(text) > 80 else ""),
                "prediction":    "uncertain",
                "confidence":    0.0,
                "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            })
            continue

        pred  = MODEL.predict([cleaned])[0]
        probs = MODEL.predict_proba([cleaned])[0]
        classes = list(MODEL.classes_)
        prob_dict = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}
        counts[pred] += 1
        output.append({
            "text":          text[:80] + ("..." if len(text) > 80 else ""),
            "prediction":    pred,
            "confidence":    round(float(max(probs)) * 100, 1),
            "probabilities": prob_dict,
        })

    return {"count": len(output), "results": output, "summary": counts}