from test_sample_tweet import PredictDisasterTweet
from logistic_regression import WithLogisticRegression
from random_forest import WithRandomForest


logistic_regression_obj = WithLogisticRegression()
logistic_regression_obj.train_model()
logistic_regression_obj.predict()


random_forest_obj = WithRandomForest()
random_forest_obj.train_model()
random_forest_obj.predict()


test_tweet = PredictDisasterTweet()
tweet = "Just happened a terrible earthquake in San Francisco!"  # Example tweet
# tweet = "I love sunny days at the beach"  # Example tweet
# tweet = "look like a bomb"  # Example tweet

prediction = test_tweet.predict(
    tweet,
    logistic_regression_obj.get_encoder(),
    logistic_regression_obj.get_model()
)
print(f"By Logistic Regression tweet is classified as: {prediction}")

prediction = test_tweet.predict(
    tweet,
    random_forest_obj.get_encoder(),
    random_forest_obj.get_model()
)
print(f"By Logistic Regression tweet is classified as: {prediction}")