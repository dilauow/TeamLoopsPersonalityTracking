import nltk
# nltk.download('omw-1.4')
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# import numpy as np
# from trainingData import TrainingData


class TextPreprocessor:
    lemmatizer = ""
    def __init__(self):

        self.lemmatizer= WordNetLemmatizer()

    def preprocess_text(self,text):
        # Tokenize text
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word.lower() not in stop_words]

        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return words


    def feedPreprocessorAnArray(self,array):
        preprocessedArray = []
        for sent in array:
            preprocessedArray.append(self.preprocess_text(sent))

        return preprocessedArray


texts = ["The quick brown fox jumps over the lazy dog.",
         "The cat in the hat.",
         "The sun is shining.",
         "It's raining cats and dogs."]

Tr = TextPreprocessor()
arr =Tr.feedPreprocessorAnArray(texts)
print(arr)