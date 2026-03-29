"""
Sentiment Analyzer - Flask Web Application
==========================================
Run:  python app.py
Open: http://127.0.0.1:5000
"""

import os
import re
import json
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_WORDS = 3      # inputs shorter than this are rejected
THRESHOLD = 55.0   # confidence below this is flagged as uncertain

# ── Load Models ────────────────────────────────────────────────────────────────
MODELS = {}


def load_models():
    global MODELS
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(model_dir):
        print("⚠️  Models not found. Run: python train_model.py")
        return

    with open(os.path.join(model_dir, "lr_pipeline.pkl"), "rb") as f:
        MODELS["lr"] = pickle.load(f)

    with open(os.path.join(model_dir, "nb_pipeline.pkl"), "rb") as f:
        MODELS["nb"] = pickle.load(f)

    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            MODELS["metrics"] = json.load(f)

    print("✅ Models loaded successfully.")


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


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400
    if not MODELS:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 500

    cleaned = clean_text(text)

    # ── Guard: reject inputs too short to analyse reliably ───────────────────
    if len(cleaned.split()) < MIN_WORDS:
        return jsonify({
            "error": f"Text too short — please enter at least {MIN_WORDS} words for a reliable prediction."
        }), 422

    results = {}
    for model_key, model_name in [("lr", "Logistic Regression"),
                                   ("nb", "Naive Bayes")]:
        pipe          = MODELS[model_key]
        prediction    = pipe.predict([cleaned])[0]
        probabilities = pipe.predict_proba([cleaned])[0]
        classes       = pipe.classes_
        prob_dict     = {cls: round(float(prob) * 100, 1)
                         for cls, prob in zip(classes, probabilities)}
        results[model_key] = {
            "model_name":    model_name,
            "prediction":    prediction,
            "confidence":    round(float(max(probabilities)) * 100, 1),
            "probabilities": prob_dict,
        }

    # ── Flag low-confidence results so the frontend can warn the user ────────
    max_conf = max(results["lr"]["confidence"], results["nb"]["confidence"])

    return jsonify({
        "text":       text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "uncertain":  max_conf < THRESHOLD,
        "results":    results,
    })


@app.route("/metrics")
def metrics():
    if "metrics" not in MODELS:
        return jsonify({"error": "Metrics not available"}), 404
    return jsonify(MODELS["metrics"])


@app.route("/batch", methods=["POST"])
def batch_analyze():
    """Analyze multiple texts at once.

    Accepts either:
      { "texts": ["text1", "text2", ...] }   <- list (used by App.jsx)
      { "texts": "text1\\ntext2" }           <- newline-separated string (legacy)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    raw = data.get("texts", [])

    if isinstance(raw, list):
        texts = [t.strip() for t in raw if isinstance(t, str) and t.strip()]
    elif isinstance(raw, str):
        if "\n" in raw:
            texts = [t.strip() for t in raw.split("\n") if t.strip()]
        else:
            texts = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        return jsonify({"error": "'texts' must be a list or a string"}), 400

    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    if len(texts) > 20:
        return jsonify({"error": "Max 20 texts per batch"}), 400
    if not MODELS:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 500

    pipe          = MODELS["lr"]
    batch_results = []
    counts        = {"positive": 0, "negative": 0, "neutral": 0}

    for text in texts:
        cleaned = clean_text(text)

        # Too-short entries → mark uncertain rather than returning a bad prediction
        if len(cleaned.split()) < MIN_WORDS:
            batch_results.append({
                "text":          text[:80] + ("..." if len(text) > 80 else ""),
                "prediction":    "uncertain",
                "confidence":    0.0,
                "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            })
            continue

        prediction    = pipe.predict([cleaned])[0]
        probabilities = pipe.predict_proba([cleaned])[0]
        classes       = pipe.classes_
        prob_dict     = {cls: round(float(prob) * 100, 1)
                         for cls, prob in zip(classes, probabilities)}
        counts[prediction] += 1
        batch_results.append({
            "text":          text[:80] + ("..." if len(text) > 80 else ""),
            "prediction":    prediction,
            "confidence":    round(float(max(probabilities)) * 100, 1),
            "probabilities": prob_dict,
        })

    return jsonify({
        "count":   len(batch_results),
        "results": batch_results,
        "summary": counts,
    })


if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5000)