from gensim.models import Word2Vec
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

# REQUIRED (only these two, safe for Streamlit)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class Word2VecRecommender:
    def __init__(self, df, description_column="Description", vector_size=100):
        self.df = df.copy()
        self.description_column = description_column
        self.vector_size = vector_size

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # 🔹 PREPROCESS TEXT
        self.df["tokens"] = self.df[self.description_column].astype(str).apply(
            self._preprocess_text
        )

        # 🔹 TRAIN WORD2VEC
        self.model = Word2Vec(
            sentences=self.df["tokens"].tolist(),
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=20
        )

        # 🔹 PRECOMPUTE VECTORS
        self.episode_vectors = np.array([
            self._sentence_vector(tokens)
            for tokens in self.df["tokens"]
        ])

    # -------------------------------
    # TEXT PREPROCESSING
    # -------------------------------
    def _preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = wordpunct_tokenize(text)  # ✅ NO punkt needed
        tokens = [w for w in tokens if w not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return tokens

    # -------------------------------
    # SENTENCE VECTOR
    # -------------------------------
    def _sentence_vector(self, tokens):
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    # -------------------------------
    # RECOMMENDATION
    # -------------------------------
    def recommend(self, user_input, top_n=5):
        user_tokens = self._preprocess_text(user_input)
        user_vector = self._sentence_vector(user_tokens).reshape(1, -1)

        scores = cosine_similarity(user_vector, self.episode_vectors)[0]
        self.df["score"] = scores

        return self.df.sort_values(by="score", ascending=False).head(top_n)
