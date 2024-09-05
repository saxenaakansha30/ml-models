from preprocess_data import PreprocessData


class PredictDisasterTweet:

    def __init__(self):
        self.preprocess_data_obj = PreprocessData()

    def predict(self, tweet, vectorizer, model):
        preprocessed_tweet = self.preprocess_data_obj.preprocess_text(tweet)
        tweet_vector = vectorizer.transform([preprocessed_tweet])
        prediction = model.predict(tweet_vector)

        return 'Disaster' if prediction[0] == 1 else 'Not a disaster'