# 🧠 SentimentAI — ML Sentiment Analyzer

> A machine learning web application that detects sentiment (positive, negative, neutral) from text using **Logistic Regression** and **Naive Bayes** classifiers with **TF-IDF** vectorization.

---

## 📌 Project Overview

This project is a full-stack ML application that:
- Trains two ML models on a real-world Twitter dataset (~24,000 samples)
- Serves predictions via a Flask web API (local) and FastAPI (Hugging Face deployment)
- Provides a beautiful interactive UI for real-time analysis
- Supports both single text and batch analysis (up to 20 texts)
- Compares two models side-by-side with confidence scores and probability bars

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Scikit-learn (Logistic Regression, Naive Bayes) |
| Feature Extraction | TF-IDF Vectorizer (unigrams + bigrams) |
| Backend (local) | Flask (Python) |
| Backend (cloud) | FastAPI on Hugging Face Spaces (Docker) |
| Frontend | React + Vite (deployed on Vercel) |
| Dataset | Kaggle Twitter Sentiment Analysis (~24,000 balanced samples) |

---

## 📁 Project Structure

```
sentiment-analyzer/
│
├── train_model.py        # ML training pipeline — run this first
├── analyze.py            # EDA + research report generator
├── app.py                # Flask web application (local)
├── hf_api.py             # FastAPI backend (Hugging Face deployment)
├── requirements.txt      # Python dependencies (Flask app)
├── hf_requirements.txt   # Python dependencies (Hugging Face)
├── Dockerfile            # Docker config for Hugging Face Spaces
│
├── templates/
│   └── index.html        # Frontend UI (Flask)
│
├── src/
│   └── App.jsx           # React frontend (Vercel deployment)
│
├── models/               # Generated after running train_model.py
│   ├── lr_pipeline.pkl
│   ├── nb_pipeline.pkl
│   └── metrics.json
│
├── twitter_training.csv  # Kaggle dataset — training split
├── twitter_validation.csv# Kaggle dataset — validation split
│
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train_model.py
```
This will:
- Load and clean the Kaggle Twitter dataset
- Balance classes to 8,000 samples each (24,000 total)
- Train Logistic Regression + Naive Bayes pipelines
- Evaluate with train/test split and 5-fold cross-validation
- Run sanity checks on common phrases
- Save models to the `models/` folder

### 3. Run the Web App
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 4. (Optional) Generate Research Report
```bash
python analyze.py
```
Outputs a full EDA + evaluation report to `analysis_report.txt`.

---

## 🤖 How It Works

### 1. Text Preprocessing
- Lowercasing
- URL and mention (`@user`) removal
- Hashtag text extraction (`#great` → `great`)
- Special character filtering
- Whitespace normalization

### 2. Feature Extraction — TF-IDF
**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical vectors:
- **TF**: How often a word appears in the document
- **IDF**: How rare/important the word is across all documents
- Uses **bigrams** (1–2 word combinations) to capture context like "not good"
- `max_features=50,000` to capture rich vocabulary from Twitter data

### 3. Models

#### Logistic Regression (Primary Model)
- Linear classifier that learns decision boundaries
- Uses feature weights (coefficients) to determine sentiment
- Better at capturing complex patterns in text
- **Typical accuracy: ~85–90%**

#### Multinomial Naive Bayes (Comparison Model)
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence (naive assumption)
- Very fast, works well as a strong baseline
- **Typical accuracy: ~80–86%**

### 4. Prediction
Both models output:
- Class label: `positive`, `negative`, or `neutral`
- Confidence score (max class probability as a percentage)
- All three class probabilities (shown as bar charts in the UI)

---

## 📊 API Endpoints

### Flask (local — `app.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/analyze` | POST | Analyze single text |
| `/batch` | POST | Analyze up to 20 texts |
| `/metrics` | GET | Model performance metrics |

### FastAPI (Hugging Face — `hf_api.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze single text |
| `/batch` | POST | Analyze up to 20 texts |
| `/metrics` | GET | CV accuracy metrics |
| `/docs` | GET | Auto-generated Swagger UI |

### Example API Call
```python
import requests

response = requests.post("http://localhost:5000/analyze", json={
    "text": "This product is absolutely amazing!"
})
print(response.json())
```

### Batch Example
```python
response = requests.post("http://localhost:5000/batch", json={
    "texts": ["Great product!", "Terrible experience.", "It was okay."]
})
print(response.json())
```

---

## 📈 Model Performance

| Metric | Logistic Regression | Naive Bayes |
|--------|-------------------|-------------|
| Test Accuracy | ~87–90% | ~81–86% |
| CV Accuracy (5-fold) | ~86–89% ± 1–2% | ~80–85% ± 2% |
| Training Speed | Fast | Very Fast |
| Interpretability | Medium (coefficients) | High (probabilities) |

> Performance figures are based on the Kaggle Twitter Sentiment dataset with 24,000 balanced samples. Actual numbers printed during `train_model.py` may vary slightly.

---

## 💡 ML Concepts Demonstrated

- ✅ Supervised Learning (Multi-class Classification)
- ✅ Real-world Dataset (Kaggle Twitter Sentiment)
- ✅ Text Preprocessing & Cleaning
- ✅ TF-IDF Feature Extraction
- ✅ N-gram Language Model (bigrams)
- ✅ Logistic Regression
- ✅ Naive Bayes Classifier
- ✅ Stratified Train/Test Split
- ✅ K-Fold Cross Validation
- ✅ Confusion Matrix
- ✅ Precision, Recall, F1 Score
- ✅ Model Comparison & Selection
- ✅ Scikit-learn Pipelines
- ✅ Model Serialization (pickle)
- ✅ REST API with Flask and FastAPI
- ✅ Docker Containerization

---

## 🔮 Future Improvements

- [ ] Add BERT/Transformer-based model for comparison
- [ ] Implement online learning with user feedback
- [ ] Add word cloud visualization
- [ ] Add confidence calibration (Platt scaling)
- [ ] Expand to 5-class fine-grained sentiment

---

## 📚 Dataset

The training dataset is the **Kaggle Twitter Sentiment Analysis** dataset:
- ~74,000 raw samples across positive, negative, neutral, and irrelevant classes
- After filtering and balancing: **8,000 samples per class (24,000 total)**
- Augmented with curated custom neutral and strongly-worded positive/negative examples
  to improve boundary clarity between classes

---

## 🛠️ Dependencies

```
flask>=2.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

For Hugging Face deployment: `fastapi`, `uvicorn`

---

## 👤 Author

**[Your Name]**
*Machine Learning Course — Pre-Mid Project*

---

## 📄 License

This project is for educational purposes.