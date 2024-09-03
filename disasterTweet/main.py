import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.data.path.append('/Users/akanksha.saxena/nltk_data')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    return ' '.join(tokens)


tfidf_vectorizer = TfidfVectorizer()


# Prepare training data:

# Apply preprocessing.
df_train = pd.read_csv('data/train.csv')
df_train['text_clean'] = df_train['text'].apply(preprocess_text)

# Feature Extraction
X_train = tfidf_vectorizer.fit_transform(df_train['text_clean'])
y_train = df_train['target']

# Prepare test Data.

# Apply preprocessing.
df_test = pd.read_csv('data/test.csv')
df_test['text_clean'] = df_test['text'].apply(preprocess_text)

# Feature Extraction
X_test = tfidf_vectorizer.transform(df_test['text_clean'])
df_submission = pd.read_csv('data/sample_submission.csv')
y_test = df_submission['target']

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Make prediction.
predictions = model.predict(X_test)
print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
print(classification_report(y_test, predictions))


def predict_disaster_tweet(tweet, vectorizer, model):
    preprocessed_tweet = preprocess_text(tweet)
    tweet_vector = vectorizer.transform([preprocessed_tweet])
    prediction = model.predict(tweet_vector)
    return 'Disaster' if prediction[0] == 1 else 'Not a disaster'


# Example usage:
tweet = "Just happened a terrible earthquake in San Francisco!"  # Example tweet
# tweet = "I love sunny days at the beach"  # Example tweet
prediction = predict_disaster_tweet(tweet, tfidf_vectorizer, model)
print(f"The tweet is classified as: {prediction}")