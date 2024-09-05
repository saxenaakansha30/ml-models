from preprocess_data import PreprocessData

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


import pandas as pd


class WithRandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=1)
        self.encoder = TfidfVectorizer()
        self.preprocess_data_ob = PreprocessData()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def prepare_training_data(self):
        df = pd.read_csv('data/train.csv')
        df['text_clean'] = df['text'].apply(self.preprocess_data_ob.preprocess_text)

        # Feature extraction.
        self.X_train = self.encoder.fit_transform(df['text_clean'])
        self.y_train = df['target']

    def prepare_testing_data(self):
        df = pd.read_csv('data/test.csv')
        df['text_clean'] = df['text'].apply(self.preprocess_data_ob.preprocess_text)
        df_target = pd.read_csv('data/sample_submission.csv')

        self.X_test = self.encoder.transform(df['text_clean'])
        self.y_test = df_target['target']

    def train_model(self):
        self.prepare_training_data()
        self.prepare_testing_data()

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        predictions = self.model.predict(self.X_test)
        # Calculate the mean absolute error of your Random Forest model on the test data
        mae_val = mean_absolute_error(predictions, self.y_test)
        print("Mean absolute error for Random Forest Model: {}".format(mae_val))

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder
