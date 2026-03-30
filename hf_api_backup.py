"""
SentimentAI — FastAPI Backend for Hugging Face Spaces
======================================================
This file runs on Hugging Face Spaces (Docker/Python SDK).
It trains the model on startup (no large file uploads needed).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
import pickle
import os
import io
import base64

# ── ML imports ─────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentAI API",
    description="Sentiment analysis using Logistic Regression & Naive Bayes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ────────────────────────────────────────────────────────────────
MIN_WORDS  = 3      # inputs shorter than this are rejected
THRESHOLD  = 55.0   # confidence below this is flagged as uncertain

# ── Dataset ─────────────────────────────────────────────────────────────────
RAW_DATA = [
    ("I absolutely love this product! It exceeded all my expectations.", "positive"),
    ("This is the best purchase I've ever made. Highly recommend!", "positive"),
    ("Amazing quality and super fast delivery. Will buy again.", "positive"),
    ("The customer service was fantastic and very helpful.", "positive"),
    ("Great value for money. Exactly what I needed.", "positive"),
    ("This app is brilliant. Makes my life so much easier!", "positive"),
    ("Beautiful design and works perfectly. Five stars!", "positive"),
    ("Outstanding performance. I'm impressed beyond words.", "positive"),
    ("Really happy with this. Works exactly as described.", "positive"),
    ("Wonderful experience from start to finish.", "positive"),
    ("The food was delicious and the service was excellent!", "positive"),
    ("I'm so glad I bought this. Pure joy to use every day.", "positive"),
    ("Top-notch quality and incredible attention to detail.", "positive"),
    ("This exceeded my expectations in every possible way.", "positive"),
    ("Superb build quality and gorgeous packaging.", "positive"),
    ("Best movie I've seen in years. Absolutely gripping.", "positive"),
    ("Feels premium and works like a charm. Love it!", "positive"),
    ("Very satisfied customer. Fast, reliable, and efficient.", "positive"),
    ("My favorite purchase this year. Cannot recommend enough!", "positive"),
    ("The team was professional, kind, and went the extra mile.", "positive"),
    ("Blown away by how good this is. Genuinely impressed.", "positive"),
    ("Works flawlessly. Setup was a breeze and it's very intuitive.", "positive"),
    ("Perfect gift idea. My friend absolutely loved it.", "positive"),
    ("Incredible battery life and stunning display.", "positive"),
    ("Such a pleasant surprise! Way better than I expected.", "positive"),
    ("Great app, easy to use, and does exactly what it promises.", "positive"),
    ("Life-changing product honestly. I use it every single day.", "positive"),
    ("Wow, just wow. I'm speechless at how good this is.", "positive"),
    ("The packaging was lovely and everything arrived intact.", "positive"),
    ("So impressed with the quality. Worth every penny!", "positive"),
    ("This is absolute garbage. Total waste of money.", "negative"),
    ("Terrible quality. Broke after two days of use.", "negative"),
    ("Worst purchase I've ever made. Do NOT buy this.", "negative"),
    ("Completely useless. Nothing works as advertised.", "negative"),
    ("I'm so frustrated. Customer support is non-existent.", "negative"),
    ("Extremely disappointed. Expected much better quality.", "negative"),
    ("Cheap, flimsy, and falls apart immediately. Avoid!", "negative"),
    ("This app crashes constantly. Ruins my productivity.", "negative"),
    ("Horrible experience. The product arrived damaged.", "negative"),
    ("Waste of time and money. Return process is a nightmare.", "negative"),
    ("False advertising. Nothing like the photos shown.", "negative"),
    ("Broke on first use. Absolutely unacceptable quality.", "negative"),
    ("The worst customer service I've ever encountered.", "negative"),
    ("Overpriced junk that doesn't do what it claims.", "negative"),
    ("Very unhappy. This ruined my plans completely.", "negative"),
    ("DO NOT BUY. I regret this purchase deeply.", "negative"),
    ("Inferior product. My old one was ten times better.", "negative"),
    ("Disgraceful quality control. Multiple defects found.", "negative"),
    ("Totally misleading description. Scam alert!", "negative"),
    ("Annoying, broken, and unreliable. A complete disaster.", "negative"),
    ("I hate this product. It made my situation worse.", "negative"),
    ("Pathetic effort from a supposedly reputable brand.", "negative"),
    ("Nothing works. Instructions are confusing and unhelpful.", "negative"),
    ("Returned immediately. Terrible in every single way.", "negative"),
    ("The battery dies after 30 minutes. Embarrassingly bad.", "negative"),
    ("Loud, slow, and ugly. Not what was described at all.", "negative"),
    ("Zero stars if I could. A complete rip-off.", "negative"),
    ("Broken out of the box. Seller refused to help. Awful.", "negative"),
    ("This is the worst thing I've bought in years. Regret.", "negative"),
    ("Extremely poor quality. Fell apart within a week.", "negative"),
    ("The product arrived on time and is as described.", "neutral"),
    ("It does what it says. Nothing more, nothing less.", "neutral"),
    ("Average quality for the price. Works okay.", "neutral"),
    ("Decent product. Has some good points and some bad.", "neutral"),
    ("It's fine. Not amazing, not terrible. Just average.", "neutral"),
    ("The delivery was on schedule and packaging was standard.", "neutral"),
    ("Works as expected. No complaints but nothing special.", "neutral"),
    ("Okay product for occasional use. Not for daily use.", "neutral"),
    ("Some features are good, others need improvement.", "neutral"),
    ("Not bad overall, though there's room for improvement.", "neutral"),
    ("It works. I've seen better, I've seen worse.", "neutral"),
    ("Standard product. Does the job when needed.", "neutral"),
    ("Acceptable for the price range. Nothing extraordinary.", "neutral"),
    ("The product is functional and does its basic job.", "neutral"),
    ("Mixed feelings. Some parts are good, others lacking.", "neutral"),
    ("Three stars. Gets the job done but nothing fancy.", "neutral"),
    ("It's adequate. I wouldn't rave about it or return it.", "neutral"),
    ("Mediocre at best. Neither impresses nor disappoints.", "neutral"),
    ("Came in standard packaging. Functions as advertised.", "neutral"),
    ("Plain and simple. Does what it needs to do.", "neutral"),
    ("Fair product for the cost. Expected a bit more though.", "neutral"),
    ("The instructions are clear and setup was straightforward.", "neutral"),
    ("Not bad for the money. Comparable to similar products.", "neutral"),
    ("Alright. I use it occasionally but nothing special.", "neutral"),
    ("Middle of the road. Average in every department.", "neutral"),
    ("Got what I paid for. No surprises either way.", "neutral"),
    ("Meets basic requirements. Not particularly impressive.", "neutral"),
    ("It's a reasonable option if you have no better choice.", "neutral"),
    ("Does the job, but I wouldn't purchase again.", "neutral"),
    ("Satisfactory. Functional design, no notable features.", "neutral"),
]

# ── Train on startup ─────────────────────────────────────────────────────────
MODEL = None
NB_MODEL = None
MODEL_METRICS = {}


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


def train():
    global MODEL, NB_MODEL, MODEL_METRICS
    print("Training models...")

    texts = [clean_text(t) for t, _ in RAW_DATA]
    labels = [l for _, l in RAW_DATA]

    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    lr_pipe.fit(texts, labels)
    nb_pipe.fit(texts, labels)

    lr_cv = cross_val_score(lr_pipe, texts, labels, cv=5).mean()
    nb_cv = cross_val_score(nb_pipe, texts, labels, cv=5).mean()

    MODEL = lr_pipe
    NB_MODEL = nb_pipe
    MODEL_METRICS = {
        "logistic_regression": {"cv_accuracy": round(float(lr_cv), 4)},
        "naive_bayes":         {"cv_accuracy": round(float(nb_cv), 4)},
        "dataset_size":        len(RAW_DATA)
    }
    print(f"✅ Models trained! LR CV: {lr_cv:.3f} | NB CV: {nb_cv:.3f}")


train()  # train on startup


# ── Schemas ──────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "SentimentAI API",
        "version": "1.0.0",
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

    # ── Guard: reject inputs too short to analyse reliably ───────────────────
    if len(cleaned.split()) < MIN_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Text too short — please enter at least {MIN_WORDS} words for a reliable prediction."
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

    # ── Flag low-confidence results so the frontend can warn the user ────────
    max_conf = max(results["lr"]["confidence"], results["nb"]["confidence"])

    return {
        "text":       body.text,
        "word_count": len(body.text.split()),
        "char_count": len(body.text),
        "uncertain":  max_conf < THRESHOLD,   # True → show warning in UI
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

        # Skip entries that are too short — mark them as uncertain instead of crashing
        if len(cleaned.split()) < MIN_WORDS:
            output.append({
                "text":        text[:80] + ("..." if len(text) > 80 else ""),
                "prediction":  "uncertain",
                "confidence":  0.0,
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