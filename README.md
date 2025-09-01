 NewsBuster is a machine learning project designed to detect and classify news articles as real or fake.
Using TF-IDF text vectorization and multiple classification models (Logistic Regression, Random Forest, Naive Bayes, Linear SVM, and MLP), the system evaluates the authenticity of news content with detailed performance reports.
📰 News Preprocessing: Cleaning and preparing text data (removing stopwords, punctuation, URLs).

🧮 TF-IDF Feature Extraction: Transforming raw text into numerical features.
🤖 Multi-Model Training: Benchmarking different ML models for fake news detection.
📊 Evaluation Metrics: Accuracy, F1-score, confusion matrix, ROC curve.
💾 Model Saving: Pre-trained models and vectorizers ready for deployment.
⚡ Prediction Function: Quick fake/real prediction for any given news article.
This repo can serve as both a learning resource for NLP beginners and a baseline system for advanced fake news detection research.
## Results

### Model Performance Table

| Model               | Accuracy | F1-score |
|--------------------|---------|----------|
| Logistic Regression | 0.987   | 0.988    |
| Random Forest       | 0.998   | 0.998    |
| Naive Bayes         | 0.935   | 0.938    |
| Linear SVM          | 0.994   | 0.995    |
| MLP                 | 0.990   | 0.990    |

> ⚠️ Note: Accuracy and F1 scores are very high, indicating the dataset might be “easy” or there may be slight data leakage. Use caution when interpreting real-world performance.

### Confusion Matrix Example

![Logistic Regression Confusion Matrix](results/LogisticRegression_confusion.png)

> Similar confusion matrices can be generated for other models using the script.
