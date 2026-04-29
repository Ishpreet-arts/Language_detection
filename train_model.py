"""
train_model.py  (v2 — improved)
--------------------------------
What changed from v1?
  OLD: CountVectorizer (word counts) + MultinomialNB
  NEW: TfidfVectorizer with CHARACTER n-grams + MultinomialNB

Why is this better?
  Word counting fails on short text like "bonjour" because
  that exact word may not be in the training vocabulary.
  Character n-grams look at small letter patterns instead —
  e.g. "bon", "onj", "njo" — and these patterns are unique
  to each language even in short sentences.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv("language.csv")
print(f"   Rows: {len(data)} | Languages: {data['language'].nunique()}")
print(f"   Missing values: {data.isnull().sum().sum()}")

# ── 2. Prepare Features & Labels ──────────────────────────────────────────────
X = np.array(data["Text"])
y = np.array(data["language"])

# ── 3. TF-IDF with Character N-Grams ──────────────────────────────────────────
# analyzer='char_wb'  -> look at CHARACTER sequences (not whole words)
# ngram_range=(1,3)   -> consider 1, 2 and 3 character sequences
# max_features=40000  -> keep only the 40,000 most useful patterns
#
# Why char n-grams?
#   "bonjour" splits into: b, o, n, j, o, u, r, bo, on, nj, jo, ou, ur, bon, onj ...
#   French always has certain character combos (like "ou", "eur", "tion")
#   that other languages don't. This works even on very short text!

print("\nBuilding TF-IDF character n-gram features...")
cv = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(1, 3),
    max_features=40000,
    use_idf=False,
    norm=None
)
X_transformed = cv.fit_transform(X)
print(f"   Feature matrix shape: {X_transformed.shape}")

# ── 4. Train/Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.20, random_state=42
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 5. Train Model ─────────────────────────────────────────────────────────────
print("\nTraining Multinomial Naive Bayes...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 7. Quick sanity check ──────────────────────────────────────────────────────
print("\nQuick sanity checks:")
sanity = [
    "Bonjour comment ca va",
    "Hello how are you today",
    "Hola como estas amigo",
    "Guten Morgen",
]
for text in sanity:
    pred = model.predict(cv.transform([text]))[0]
    print(f"   '{text}' --> {pred}")

# ── 8. Save ────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f)

print("\nSaved model.pkl and vectorizer.pkl")
print("Run: streamlit run app.py")
