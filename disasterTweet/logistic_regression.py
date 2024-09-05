import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

from preprocess_data import PreprocessData


class WithLogisticRegression:

    tfidf_vectorizer = None
    model = None

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.preprocess_obj = PreprocessData()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def prepare_train_data(self):
        # Apply preprocessing.
        df_train = pd.read_csv('data/train.csv')
        df_train['text_clean'] = df_train['text'].apply(self.preprocess_obj.preprocess_text)

        # Feature Extraction
        self.X_train = self.tfidf_vectorizer.fit_transform(df_train['text_clean'])
        self.y_train = df_train['target']

    def prepare_test_data(self):
        # Apply preprocessing.
        df_test = pd.read_csv('data/test.csv')
        df_test['text_clean'] = df_test['text'].apply(self.preprocess_obj.preprocess_text)

        # Feature Extraction
        self.X_test = self.tfidf_vectorizer.transform(df_test['text_clean'])
        df_submission = pd.read_csv('data/sample_submission.csv')
        self.y_test = df_submission['target']

    def train_model(self):
        # Model training
        self.prepare_train_data()
        self.prepare_test_data()

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # Make prediction.
        predictions = self.model.predict(self.X_test)
        mae_val = mean_absolute_error(predictions, self.y_test)
        print("Mean absolute error for Logistic Regression: {}".format(mae_val))

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.tfidf_vectorizer