from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TfidfRecommender:
    def __init__(self, df, text_column="Description"):
        self.df = df.copy()
        self.text_column = text_column
        # self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.episode_vectors = self.vectorizer.fit_transform(self.df[self.text_column])

    def recommend(self, user_input, top_n=5):
        user_vector = self.vectorizer.transform([user_input])
        scores = cosine_similarity(user_vector, self.episode_vectors)[0]

        # Filter by threshold
        self.df["score"] = scores
        # filtered = self.df[self.df["score"] >= self.threshold]
        return self.df.sort_values(by="score", ascending=False).head(top_n)
