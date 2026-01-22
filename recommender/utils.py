import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(user_vec, episode_vecs):
    return cosine_similarity(user_vec, episode_vecs)[0]

def episode_score(user_words, episode_text, model):
    ep_words = episode_text.lower().split()
    ep_words = [w for w in ep_words if w in model.wv]

    if not user_words or not ep_words:
        return 0.0

    user_vecs = np.array([model.wv[w] for w in user_words])
    ep_vecs = np.array([model.wv[w] for w in ep_words])

    sim_matrix = cosine_similarity(user_vecs, ep_vecs)

    # Take max similarity for each user word
    max_per_word = sim_matrix.max(axis=1)

    # Final episode score
    return max_per_word.mean()
