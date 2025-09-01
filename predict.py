import joblib
from train_fake_news import clean_text, predict_news

tfidf = joblib.load("models/tfidf_vectorizer.joblib")
lr = joblib.load("models/logreg_model.joblib")

while True:
    news = input("Haber metni: ")
    if news.lower() == 'q':
        break
    result = predict_news(news, model=lr, vectorizer=tfidf)
    print(f"Tahmin: {result['label'].upper()}, Sahte Olma Olasılığı: {result['probability_fake']:.2f}\n")
