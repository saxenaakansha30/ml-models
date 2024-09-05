import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import string

nltk.data.path.append('/Users/akanksha.saxena/nltk_data')


class PreprocessData:

    def __init__(self):
        pass

    def preprocess_text(self, text) -> str:
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Join tokens back into a single string
        return ' '.join(tokens)