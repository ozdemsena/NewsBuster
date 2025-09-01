# train_fake_news.py
import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
import joblib
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# -----------------------
# 1) Veri yükleme
# -----------------------
def load_kaggle_fake_real(path='data'):
    fake_path = os.path.join(path, 'Fake.csv')
    true_path = os.path.join(path, 'True.csv')
    assert os.path.exists(fake_path), f"{fake_path} yok!"
    assert os.path.exists(true_path), f"{true_path} yok!"
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 'fake'
    df_true['label'] = 'real'
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # karıştır
    return df

df = load_kaggle_fake_real('data')
print('Toplam örnek:', len(df))
print(df[['label']].value_counts())

# -----------------------
# 2) Temizleme
# -----------------------
def combine_text(row):
    title = '' if pd.isna(row.get('title')) else str(row.get('title'))
    text  = '' if pd.isna(row.get('text'))  else str(row.get('text'))
    return (title + ' ' + text).strip()

df['content'] = df.apply(combine_text, axis=1)
df = df[df['content'].str.len() > 10].reset_index(drop=True)

STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # url çıkar
    text = re.sub(r'<.*?>', ' ', text)              # html çıkar
    text = re.sub(r'[^a-z\s]', ' ', text)           # harf dışı çıkar
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

df['clean'] = df['content'].map(clean_text)

# -----------------------
# 3) Train/Test split
# -----------------------
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

X_train_text, y_train = df_train['clean'], df_train['label'].map({'real':0,'fake':1})
X_test_text,  y_test  = df_test['clean'],  df_test['label'].map({'real':0,'fake':1})

# -----------------------
# 4) TF-IDF vektörleştirme
# -----------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,1), stop_words='english')
tfidf.fit(X_train_text)

X_train = tfidf.transform(X_train_text)
X_test = tfidf.transform(X_test_text)

# -----------------------
# 5) Modeller
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# -----------------------
# 6) Değerlendirme
# -----------------------
results = []
for name, model in models.items():
    print(f"\n=== Eğitim: {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1))
    print(classification_report(y_test, y_pred, target_names=['real','fake']))

results_df = pd.DataFrame(results, columns=["Model","Accuracy","F1"])
print("\n--- Sonuç Tablosu ---")
print(results_df)

# -----------------------
# 7) Model ve vectorizer kaydet
# -----------------------
os.makedirs('models', exist_ok=True)
joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
joblib.dump(models["Logistic Regression"], 'models/logreg_model.joblib')

# -----------------------
# 8) Tahmin fonksiyonu
# -----------------------
def predict_news(text, model=models["Logistic Regression"], vectorizer=tfidf, threshold=0.5):
    txt = clean_text(text)
    x = vectorizer.transform([txt])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(x)[0][1]
        label = 'fake' if proba >= threshold else 'real'
        return {'label': label, 'probability_fake': float(proba)}
    else:
        pred = model.predict(x)[0]
        return {'label': 'fake' if pred==1 else 'real', 'probability_fake': None}

# Örnek kullanım
example = "The president announced a new plan to reduce taxes next year."
print("\nTahmin örneği:", predict_news(example))
