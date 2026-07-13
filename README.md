# 🎬 Sentiment Analysis — NLP Pipeline

> Classifying IMDB movie reviews as positive or negative using classical NLP and transformer-based approaches — with interpretable feature analysis and production trade-off documentation.

---

## 📌 Problem Statement

Sentiment analysis is a foundational NLP task with direct business applications — product review monitoring, customer support triage, brand reputation tracking, and churn signal detection. This project builds a full text classification pipeline, comparing classical ML approaches with modern transformer-based models.

Business question: Can we automatically classify whether a customer review is positive or negative — and which words drive that classification?

---

## 📂 Dataset

- Source: [IMDB Dataset of 50K Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Rows:50,000 reviews
- Columns: `review` (raw text), `sentiment` (positive/negative)
- Class balance: Perfectly balanced — 25,000 positive / 25,000 negative
- Advantage over churn project: No class imbalance handling needed (SMOTE not required)

---

## 🗂️ Project Structure

sentiment-analysis-nlp/
│
├── notebook.ipynb                    ← Full pipeline: EDA, preprocessing, modeling, BERT
├── README.md                         ← Project overview (you are here)
├── requirements.txt                  ← Python dependencies
│
└── images/
    ├── review_length_dist.png            ← EDA: review length by sentiment
    ├── top_words_by_sentiment.png        ← Top 15 words per sentiment class
    └── confusion_matrix_lr.png           ← Logistic Regression confusion matrix
---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Confirmed perfect 50/50 class balance — no imbalance handling required
- Analyzed review length distribution — negative reviews trend slightly longer on average
- Extracted top words before and after cleaning — confirmed stopword removal was necessary (pre-cleaning top words: "the", "and", "of"; post-cleaning: meaningful sentiment words)
- Identified HTML residue (`<br />` tags) from web scraping that required targeted regex cleaning

### 2. Text Preprocessing Pipeline
Custom cleaning function applied to all 50,000 reviews:

| Step | Method | Reason |
|---|---|---|
| HTML removal | `re.sub(r'<.*?>', ' ', text)` | IMDB data scraped from web — contains `<br />` tags |
| Lowercasing | `.lower()` | Ensures "Great" and "great" are treated as same word |
| Punctuation removal | `re.sub(r'[^a-z\s]', '', text)` | Removes noise characters |
| Stopword removal | NLTK stopwords (~180 words) | Removes "the", "is", "at" — no sentiment signal |

### 3. TF-IDF Vectorization
- Converted cleaned text to numerical vectors using TF-IDF
- `max_features=5000` — top 5,000 most informative words only
- `min_df=5` — ignores words appearing in fewer than 5 reviews (likely typos/rare names)
- `max_df=0.8` — ignores words appearing in 80%+ of reviews (too generic, e.g., "movie", "film")
- Fit **only on training data**, then transformed test set — preventing vocabulary leakage

### 4. Models Trained & Compared

| Model | Accuracy | F1-Score |
|---|---|---|
| Multinomial Naive Bayes | 85.37% | 0.8546 |
| **Logistic Regression** | **88.94%** | **~0.89 ✅ winner** |
| DistilBERT (pretrained) | ~93%* | benchmark |

*DistilBERT SST-2 benchmark on IMDB — demonstrated on sample predictions, not fine-tuned in this project

### 5. Model Interpretability — Top Predictive Words
Extracted Logistic Regression coefficients to identify which words drive each sentiment:

**Top positive words:** great (7.13), excellent (6.73), perfect (5.31), amazing (4.97), best (4.84), loved (4.61), hilarious (4.52), wonderful (4.47)

**Top negative words:** worst (-10.29), waste (-8.05), awful (-7.59), bad (-7.44), boring (-6.08), poor (-5.57), terrible (-5.50), disappointing (-4.58)

> This coefficient-based interpretability is the classical NLP equivalent of SHAP for tree models — directly showing which features drive predictions and in which direction.

### 6. Classical vs Transformer Trade-off Analysis

| | TF-IDF + Logistic Regression | DistilBERT |
|---|---|---|
| Accuracy (IMDB) | 88.94% | ~93% |
| Training time | < 30 seconds | Hours (GPU recommended) |
| Inference speed | Milliseconds | ~100ms per sample |
| Interpretability | High (coefficient weights) | Low (black box) |
| Compute cost | Minimal (CPU) | High (GPU) |
| Best for | Production at scale, low latency | Highest accuracy tasks |

**Production recommendation:** For a real-time review classification system where latency and infrastructure cost matter, TF-IDF + Logistic Regression at 88.94% is the right engineering choice. DistilBERT/BERT fine-tuning is justified when 3-4% accuracy gain outweighs the compute and deployment overhead.

---

## 📊 Key Visualizations

### Top 15 words by sentiment (model coefficients)
![Top Words](images/top_words_by_sentiment.png)

### Confusion Matrix — Logistic Regression
![Confusion Matrix](images/confusion_matrix_lr.png)

---

## 💡 Key Insights

1. **TF-IDF + Logistic Regression achieves 88.94% accuracy** — competitive with much heavier approaches, at a fraction of the compute cost

2. **"Worst" is the single strongest negative predictor** (coefficient: -10.29) — 44% stronger than the next negative word "waste" (-8.05), suggesting extreme language is the clearest negative signal

3. **Negative reviews are slightly longer on average** — people tend to elaborate more when expressing dissatisfaction, a pattern visible in the review length distribution

4. **Pretrained DistilBERT without fine-tuning misclassifies nuanced reviews** — 3/5 correct on sample predictions vs 88.94% for LR, showing that domain-specific fine-tuning is critical for transformer models to outperform classical approaches

5. **Balanced datasets simplify the pipeline significantly** — unlike the churn project (73/27 split requiring SMOTE), a balanced dataset allows accuracy as a reliable primary metric and removes imbalance handling from the pipeline entirely

---

## 🐛 Debugging Note

During initial training runs, models scored ~54% accuracy — barely above random guessing on a 50/50 dataset. Root cause: running Colab cells out of order caused `X_train_tfidf`/`X_test_tfidf` and `y_train`/`y_test` to correspond to different random splits of the data, even though each individually appeared correct. Fix: re-running the full pipeline (split → vectorize → train → evaluate) as one contiguous block in a fresh session. Lesson: always validate that your feature matrix and label vector correspond to the same data split before interpreting results.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Text processing | NLTK, regex |
| Feature extraction | scikit-learn TfidfVectorizer |
| Modeling | scikit-learn (Logistic Regression, Naive Bayes) |
| Transformer NLP | HuggingFace Transformers (DistilBERT) |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab |

---

## ▶️ How to Run

1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
3. Upload the CSV to Colab:
```python
from google.colab import files
files.upload()
```
4. Run all cells top to bottom in a single session

---

## 📝 Learnings & Challenges

- **Stale Colab variables caused near-random accuracy (54%)** — feature matrix and labels corresponded to different random splits. Re-running the full pipeline as one block resolved it. Critical lesson: always run split → vectorize → train as one atomic block.

- **HTML residue in scraped data** — `<br />` tags from IMDB's website leaked into text. Standard text cleaning missed the "br" residue — caught and removed with targeted regex.

- **TF-IDF parameter choices matter significantly** — `max_df=0.8` filtering domain-generic words like "movie" and "film" improved signal quality meaningfully. Without it, the most common words would dominate the feature space despite carrying no sentiment information.

- **Pretrained ≠ fine-tuned** — DistilBERT without domain-specific fine-tuning performs worse than a well-tuned classical model on this task. This reinforces that model selection is a engineering trade-off, not just a "bigger is better" decision.

---

## requirements.txt

```
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
nltk==3.8.1
transformers==4.35.0
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 🔗 Connect

**[Deepika J]**  
[LinkedIn](https://www.linkedin.com/in/deepika-jalamanchili/) · [GitHub](https://github.com/Deepika-J2106) · [deepikajalamanchili.9999@gmail.com](mailto:deepikajalamanchili.9999@gmail.com)
