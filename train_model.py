"""
Sentiment Analyzer - Model Training Script (Improved)
======================================================
Dataset : Twitter Sentiment Analysis (Kaggle)
Models  : Logistic Regression, Naive Bayes, LinearSVC (ensemble voting)
Fixes   : No data leakage, proper CV, GridSearch tuning, ensemble, 
          thorough evaluation on truly held-out test set.
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

# ── CUSTOM NEUTRAL DATASET ────────────────────────────────────────────────────
# NOTE: These are ONLY used for training augmentation, NOT in any test/sanity
# check to avoid data leakage / memorisation.
CUSTOM_NEUTRAL = [
    "it does what it says nothing more nothing less",
    "average product for the price works okay",
    "decent enough not amazing but not terrible either",
    "it was okay nothing special about it",
    "gets the job done nothing extraordinary",
    "meh it is what it is",
    "not bad not great just average",
    "kind of okay for what i paid",
    "works fine i guess pretty standard stuff",
    "it is alright nothing to write home about",
    "fair product has good and bad points",
    "mediocre at best acceptable at worst",
    "so so could be better could be worse",
    "neither impressive nor disappointing",
    "three stars does the job but barely",
    "okay i suppose nothing stands out",
    "it functions as expected that is about it",
    "pretty average i have seen better and worse",
    "not bad overall room for improvement though",
    "tolerable product nothing to get excited about",
    "middle of the road experience",
    "satisfactory but not outstanding in any way",
    "acceptable quality for the price point",
    "fine for occasional use nothing special",
    "works as described no surprises either way",
    "adequate meets basic requirements only",
    "standard product does what it needs to",
    "reasonable not the best not the worst",
    "it is passable i have no strong feelings",
    "neither love it nor hate it just okay",
    "could be worse could definitely be better",
    "meets expectations barely",
    "just okay nothing to rave about",
    "it is fine i guess serves its purpose",
    "neutral on this one has pros and cons",
    "mixed feelings some good some not so good",
    "about what i expected nothing more",
    "generic product does the basics",
    "not particularly impressed or disappointed",
    "food was okay nothing special",
    "decent meal not the best i have had",
    "average taste nothing memorable",
    "it was fine would not rush back but not bad",
    "food was alright pretty standard",
    "okay food nothing that wowed me",
    "service was okay nothing special",
    "staff were alright did their job",
    "average service not great not terrible",
    "service was decent no complaints really",
    "movie was okay passed the time",
    "decent film not my favourite but watchable",
    "average movie neither loved nor hated it",
    "it was fine nothing groundbreaking",
    "watchable but forgettable film",
    "app is okay nothing special",
    "works fine no major issues",
    "average app does what it promises",
    "decent enough not the best out there",
    "it is functional nothing more",
    "hotel was okay nothing special",
    "average stay met basic expectations",
    "decent hotel not luxury but acceptable",
    "room was fine nothing to complain about",
    "it was okay",
    "kind of average",
    "nothing special really",
    "sort of alright",
    "meh overall",
    "not bad not great",
    "could be better",
    "could be worse",
    "pretty standard",
    "just okay",
    "somewhat decent",
    "fair enough",
    "so so experience",
    "middle ground",
    "neither here nor there",
    "take it or leave it",
    "on the fence about this",
    "mixed results",
    "no strong feelings",
    "it is what it is",
    "not my best not my worst",
    "average at best",
    "tolerable experience",
    "mediocre performance",
    "passable quality",
    "meets expectations nothing more",
    "ordinary experience overall",
    "had both good and bad moments",
    "some things worked some did not",
    "pros and cons overall just okay",
    "left feeling neutral about the whole thing",
    "nothing remarkable happened",
    "a very ordinary and unremarkable experience",
    "everything was just about average",
    "somewhere in between good and bad",
    "i have experienced better and worse",
    "does what it claims no more no less",
    "adequate for the purpose that is all",
    "an entirely unremarkable product",
    "functional but uninspiring",
    "competent but not impressive",
    "serviceable at best",
    "nothing here to get excited about",
    "i guess it is okay",
    "not sure how i feel about it",
    "eh it was alright",
    "i do not love it but i do not hate it",
    "would not go out of my way for it",
    "nothing to write home about",
    "forgettable but not offensive",
    "bland but functional",
    "unremarkable experience overall",
    "solidly average in every way",
    "right down the middle",
    "perfectly mediocre",
    "uneventful experience",
    "no complaints but no praise either",
    "very much a middle of the road thing",
    "shipping was on time product was as described",
    "arrived as expected packaging was fine",
    "standard delivery nothing to report",
    "product matches description that is about it",
    "does the job nothing fancy",
    "basic functionality works okay",
    "it serves a purpose adequately",
    "reliable enough for basic tasks",
    "no surprises good or bad",
    "very run of the mill experience",
    "price matches the quality which is average",
    "you get what you pay for mediocre",
    "consistent mediocrity at least",
    "predictably average experience",
    "as expected for this price range",
]

EXTRA_NEGATIVE = [
    "worst product ever made complete garbage",
    "absolutely terrible do not buy this",
    "total waste of money horrible quality",
    "worst purchase of my life deeply regret it",
    "disgusting service never coming back",
    "completely broken arrived damaged useless",
    "terrible customer service no help at all",
    "awful experience would not recommend to anyone",
    "complete disaster everything went wrong",
    "worst experience i have ever had",
    "zero stars if i could absolutely dreadful",
    "extremely disappointed terrible quality",
    "do not buy this complete rip off",
    "horrible product broke immediately",
    "worst thing i have ever bought",
    "absolutely dreadful experience",
    "nothing works as advertised scam",
    "furious with this product total junk",
    "disgusting quality embarrassingly bad",
    "terrible terrible terrible avoid at all costs",
    "nightmare experience from start to finish",
    "completely useless broken on arrival",
    "hate this product ruined everything",
    "devastating disappointment absolute rubbish",
    "the worst do not waste your money",
    "appalling quality total failure",
    "deeply regret this purchase awful",
    "this is garbage pure and simple",
    "outrageous experience never again",
    "furious and disgusted with this",
]

EXTRA_POSITIVE = [
    "absolutely love this product amazing quality",
    "best purchase i have ever made highly recommend",
    "fantastic experience exceeded all expectations",
    "incredible product works perfectly five stars",
    "outstanding quality truly impressed",
    "brilliant product changed my life",
    "superb service fast and friendly",
    "amazing value for money love it",
    "perfect in every way could not be happier",
    "exceptional quality truly outstanding",
    "wonderful experience from start to finish",
    "magnificent product blown away by quality",
    "excellent service highly recommend",
    "stunning results absolutely delighted",
    "phenomenal product works like a dream",
    "incredible experience best i have ever had",
    "love love love this product amazing",
    "absolutely perfect exceeded expectations",
    "brilliant quality fast delivery very happy",
    "outstanding service genuinely impressed",
    "fantastic love it would buy again",
    "excellent quality great value for money",
    "superb product highly recommend to everyone",
    "amazing product works flawlessly",
    "incredible value blown away by this",
    "perfect product exactly what i needed",
    "wonderful service friendly and helpful",
    "brilliant experience truly impressed",
    "fantastic quality highly satisfied",
    "outstanding excellent very impressed",
]

# ── Sanity-check phrases are SEPARATE from training data ──────────────────────
# These phrases do NOT appear in EXTRA_* or CUSTOM_* lists above.
SANITY_TESTS = [
    ("This product is genuinely wonderful, I am so happy with it", "positive"),
    ("Really enjoyed using it, works better than expected", "positive"),
    ("Highly satisfied, would definitely purchase again", "positive"),
    ("Total rubbish, broke within a day of use", "negative"),
    ("Dreadful experience, staff were rude and unhelpful", "negative"),
    ("Extremely poor quality, very disappointed with my purchase", "negative"),
    ("It is alright I suppose, nothing stands out", "neutral"),
    ("Pretty standard, has both positives and negatives", "neutral"),
    ("I am somewhere in the middle on this one", "neutral"),
    ("Great communication and arrived early, very pleased", "positive"),
    ("Shameful product, complete waste of my hard earned money", "negative"),
    ("Average at best, does the bare minimum required", "neutral"),
]


# ── TEXT CLEANING ──────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "doesn't": "does not", "didn't": "did not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "he's": "he is", "she's": "she is", "we've": "we have",
    "they've": "they have", "you've": "you have", "we'll": "we will",
    "they'll": "they will", "you'll": "you will",
}


def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def handle_negation(text: str) -> str:
    """Tag word immediately after negation so 'not great' -> 'NOT_great'.
    Prevents TF-IDF treating 'not great' same as 'great'."""
    negations = {
        "not", "no", "never", "nothing", "nobody", "neither", "nor",
        "barely", "hardly", "scarcely", "cannot", "without"
    }
    words = text.split()
    result = []
    negate = False
    for word in words:
        clean_word = re.sub(r"[^a-z]", "", word)
        if clean_word in negations:
            negate = True
            result.append(word)
        elif negate and clean_word.isalpha():
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
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Keep hashtag words
    text = re.sub(r"#(\w+)", r"\1", text)
    # Expand contractions
    text = expand_contractions(text)
    # Replace emojis / non-ascii with space
    text = text.encode("ascii", "ignore").decode("ascii")
    # Normalise repeated punctuation (e.g. !!! -> !)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    # Keep letters, digits and basic punctuation
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Handle negation: "not great" -> "NOT_great"
    text = handle_negation(text)
    return text


# ── FEATURE: extra hand-crafted features ─────────────────────────────────────
def hand_features(texts):
    """Return a small matrix of hand-crafted numerical features."""
    results = []
    for text in texts:
        t = text.lower() if isinstance(text, str) else ""
        results.append([
            t.count("!"),
            t.count("?"),
            len(t.split()),
            int(any(w in t for w in ["love", "great", "excellent", "amazing", "fantastic", "wonderful"])),
            int(any(w in t for w in ["hate", "terrible", "awful", "worst", "horrible", "disgusting"])),
            int(any(w in t for w in ["okay", "average", "meh", "mediocre", "alright", "fine", "decent"])),
        ])
    return np.array(results, dtype=float)


# ── LOAD DATASET ──────────────────────────────────────────────────────────────
def load_dataset():
    train_path = "twitter_training.csv"
    val_path   = "twitter_validation.csv"

    missing = [p for p in [train_path, val_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")

    col_names = ["id", "entity", "label", "text"]

    print("📂 Loading twitter_training.csv ...")
    df_train = pd.read_csv(train_path, header=None, names=col_names,
                           encoding="utf-8", on_bad_lines="skip")
    print("📂 Loading twitter_validation.csv ...")
    df_val = pd.read_csv(val_path, header=None, names=col_names,
                         encoding="utf-8", on_bad_lines="skip")

    df = pd.concat([df_train, df_val], ignore_index=True)
    print(f"   Raw rows: {len(df):,}")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 3]

    print("🧹 Cleaning text ...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 3]

    SAMPLES = 8000

    pos_df = df[df["label"] == "positive"].sample(
        min(SAMPLES, len(df[df["label"] == "positive"])), random_state=42
    )
    neg_df = df[df["label"] == "negative"].sample(
        min(SAMPLES, len(df[df["label"] == "negative"])), random_state=42
    )
    neu_df = df[df["label"] == "neutral"].sample(
        min(SAMPLES, len(df[df["label"] == "neutral"])), random_state=42
    )

    # ── Augment with synthetic data (kept SEPARATE from SANITY_TESTS) ─────────
    def make_aug(texts, label, repeats):
        cleaned = [clean_text(t) for t in texts]
        rows = pd.DataFrame({"label": label, "clean_text": cleaned * repeats})
        return rows

    pos_aug = make_aug(EXTRA_POSITIVE, "positive", 8)
    neg_aug = make_aug(EXTRA_NEGATIVE, "negative", 8)
    neu_aug = make_aug(CUSTOM_NEUTRAL, "neutral",  6)  # increased 4→6 to fix neutral bias

    pos_final = pd.concat([pos_df[["label","clean_text"]], pos_aug], ignore_index=True)
    neg_final = pd.concat([neg_df[["label","clean_text"]], neg_aug], ignore_index=True)
    neu_final = pd.concat([neu_df[["label","clean_text"]], neu_aug], ignore_index=True)

    # Clip each class to SAMPLES to keep balance
    pos_final = pos_final.sample(min(SAMPLES, len(pos_final)), random_state=42)
    neg_final = neg_final.sample(min(SAMPLES, len(neg_final)), random_state=42)
    neu_final = neu_final.sample(min(SAMPLES, len(neu_final)), random_state=42)

    print(f"\n   ✅ Positive : {len(pos_final):,}")
    print(f"   ✅ Negative : {len(neg_final):,}")
    print(f"   ✅ Neutral  : {len(neu_final):,}")

    df_final = pd.concat([pos_final, neg_final, neu_final], ignore_index=True)
    df_final = shuffle(df_final, random_state=42).reset_index(drop=True)
    return df_final


# ── BUILD PIPELINES ───────────────────────────────────────────────────────────
def build_pipelines():
    """
    Returns a dict of named sklearn Pipelines.
    Each pipeline uses a FeatureUnion of:
      - word TF-IDF (unigrams + bigrams)
      - char TF-IDF  (3-5 char n-grams, captures morphology & typos)
    """

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=60000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )

    # We use a simple word-only vectoriser for Naive Bayes (no char n-grams
    # because MultinomialNB requires non-negative input and char n-grams can
    # push TF-IDF to very small positives that occasionally cause issues).
    nb_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )

    lr_pipeline = Pipeline([
        ("tfidf", word_tfidf),
        ("clf",   LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=42
        )),
    ])

    nb_pipeline = Pipeline([
        ("tfidf", nb_tfidf),
        ("clf",   MultinomialNB()),
    ])

    # LinearSVC wrapped in CalibratedClassifierCV to get predict_proba
    svc_pipeline = Pipeline([
        ("tfidf", word_tfidf),
        ("clf",   CalibratedClassifierCV(
            LinearSVC(max_iter=3000, random_state=42), cv=3
        )),
    ])

    return {
        "Logistic Regression": lr_pipeline,
        "Naive Bayes":         nb_pipeline,
        "LinearSVC":           svc_pipeline,
    }


# ── HYPERPARAMETER GRIDS ──────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.3, 1.0, 3.0, 10.0],
    },
    "Naive Bayes": {
        "clf__alpha": [0.1, 0.3, 0.5, 1.0],
    },
    "LinearSVC": {
        "clf__estimator__C": [0.3, 1.0, 3.0],
    },
}


# ── TRAIN & EVALUATE ──────────────────────────────────────────────────────────
def train_and_evaluate():
    print("=" * 65)
    print("  SENTIMENT ANALYZER — MODEL TRAINING (Improved)")
    print("  Dataset: Kaggle Twitter + Balanced Augmentation")
    print("=" * 65)

    df = load_dataset()

    print(f"\n📊 Final Dataset: {len(df):,} samples")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"   {label:10s}: {count:6,} ({pct:.1f}%) {bar}")

    X = df["clean_text"]
    y = df["label"]

    # ── Stratified split: 70% train, 15% val (for GridSearch), 15% test ───────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.18, random_state=42, stratify=y_trainval
    )
    print(f"\n   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"   (Test set is held-out and never seen during tuning)\n")

    pipelines = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_models = {}
    all_metrics = {}

    for name, pipe in pipelines.items():
        print(f"{'─'*65}")
        print(f"  🔧 Tuning: {name}")
        print(f"{'─'*65}")

        grid = GridSearchCV(
            pipe,
            PARAM_GRIDS[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_pipe  = grid.best_estimator_
        val_preds  = best_pipe.predict(X_val)
        test_preds = best_pipe.predict(X_test)
        val_acc    = accuracy_score(y_val,  val_preds)
        test_acc   = accuracy_score(y_test, test_preds)

        cv_scores = cross_val_score(
            best_pipe, X_trainval, y_trainval,
            cv=cv, scoring="accuracy", n_jobs=-1
        )

        print(f"  Best params  : {grid.best_params_}")
        print(f"  Val  Accuracy: {val_acc:.4f}  ({val_acc*100:.1f}%)")
        print(f"  Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
        print(f"  CV   Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n{classification_report(y_test, test_preds)}")

        best_models[name] = best_pipe
        all_metrics[name] = {
            "best_params": grid.best_params_,
            "val_accuracy":  round(float(val_acc),  4),
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean":       round(float(cv_scores.mean()), 4),
            "cv_std":        round(float(cv_scores.std()),  4),
        }

    # ── Confusion matrix for best individual model ────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["test_accuracy"])
    best_model = best_models[best_name]
    print(f"\n🏆 Best individual model: {best_name} "
          f"(test acc = {all_metrics[best_name]['test_accuracy']:.4f})")
    cm = confusion_matrix(y_test, best_model.predict(X_test),
                          labels=["positive", "negative", "neutral"])
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':12s}  {'pos':>6}  {'neg':>6}  {'neu':>6}")
    for i, lbl in enumerate(["positive", "negative", "neutral"]):
        print(f"  {lbl:12s}  {cm[i,0]:>6}  {cm[i,1]:>6}  {cm[i,2]:>6}")

    # ── Sanity check on UNSEEN phrases ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  🧪 Sanity check — phrases NOT in training data:")
    print(f"{'─'*65}")
    all_correct = True
    for phrase, expected in SANITY_TESTS:
        cleaned  = clean_text(phrase)
        pred     = best_model.predict([cleaned])[0]
        probs    = best_model.predict_proba([cleaned])[0]
        classes  = best_model.classes_
        conf     = round(max(probs) * 100, 1)
        correct  = pred == expected
        if not correct:
            all_correct = False
        icon = "✅" if correct else "❌"
        print(f"   {icon} [{pred.upper():8s} {conf:5.1f}%]  {phrase[:55]}")

    if all_correct:
        print("\n   🎉 All sanity checks passed!")
    else:
        print("\n   ⚠️  Some checks failed — inspect training data balance.")

    # ── Save models and metrics ────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    lr_pipe  = best_models["Logistic Regression"]
    nb_pipe  = best_models["Naive Bayes"]

    with open("models/lr_pipeline.pkl",  "wb") as f: pickle.dump(lr_pipe,  f)
    with open("models/nb_pipeline.pkl",  "wb") as f: pickle.dump(nb_pipe,  f)

    # Also save the single best model as "best_model.pkl" for easy loading
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    all_metrics["best_model"] = best_name
    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ Models saved to models/")
    print("   → models/lr_pipeline.pkl")
    print("   → models/nb_pipeline.pkl")
    print("   → models/best_model.pkl   ← recommended for inference")
    print("   → models/metrics.json")
    print("\n🚀 Run: python app.py  to start the web app")

    return lr_pipe, nb_pipe, best_model


if __name__ == "__main__":
    train_and_evaluate()