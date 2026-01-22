import streamlit as st
import pandas as pd
from recommender.word2vec_recommender import Word2VecRecommender
from recommender.tfidf_recommender import TfidfRecommender

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TMKOC Episode Recommender",
    page_icon="📺",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
/* PAGE TITLE */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #d32f2f;
    margin-bottom: 20px;
}

/* SECTION CONTAINER - REMOVE WHITE BOX */
.section {
    background-color: transparent;   /* make transparent */
    padding: 0;                      /* remove extra padding */
    border-radius: 0;
    box-shadow: none;                 /* remove shadow */
    margin-bottom: 20px;
}

/* EPISODE DESCRIPTION */
.description {
    font-size: 20px;
    margin-bottom: 8px;
}

/* LINKS */
.link {
    font-size: 18px;
    color: #1a73e8;
    margin-right: 10px;
}

/* SCORE */
.score {
    color: #2e7d32;
    font-weight: bold;
    font-size: 18px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">TMKOC Episode Recommendation System</div>', unsafe_allow_html=True)

st.image("images/tmkoc.png", width=700)  # use width to control image size

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/TMKOC_clean_data.csv")

df = load_data()

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models(df):
    tfidf = TfidfRecommender(df)
    w2v = Word2VecRecommender(df)
    return tfidf, w2v

tfidf_model, w2v_model = load_models(df)

# ---------------- USER INPUT ----------------
user_input = st.text_input(
    "🔍 Enter keyword or description",
    placeholder="e.g. marriage, tax problem, society comedy"
)

top_n = st.slider("Number of results", 1, 10, 5)

# ---------------- BUTTON ----------------
if st.button("🎯 Recommend Episodes"):
    if not user_input.strip():
        st.warning("Please enter some text")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔎 Keyword Match Based (TF-IDF)")
            tfidf_results = tfidf_model.recommend(user_input, top_n)

            for _, row in tfidf_results.iterrows():
                st.markdown(f"<p class='description'>**Episode:** {row.get('Episode_No','')} - {row['Episode_title']}</p>", unsafe_allow_html=True)
                if "sonyliv_link" in row and pd.notna(row["sonyliv_link"]):
                    st.markdown(f"<a class='link' href='{row['sonyliv_link']}' target='_blank'>SonyLiv Link</a>", unsafe_allow_html=True)
                if "yt_link" in row and pd.notna(row["yt_link"]):
                    st.markdown(f"<a class='link' href='{row['yt_link']}' target='_blank'>YouTube Link</a>", unsafe_allow_html=True)
                st.markdown(f"<span class='score'>Score: {row['score']:.3f}</span>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

        # WORD2VEC RESULTS
        with col2:
            st.subheader("🧠 Semantic Meaning Based (Word2Vec)")
            w2v_results = w2v_model.recommend(user_input, top_n)

            for _, row in w2v_results.iterrows():
                st.markdown(f"<p class='description'>**Episode:** {row.get('Episode_No','')} - {row['Episode_title']}</p>", unsafe_allow_html=True)
                if "sonyliv_link" in row and pd.notna(row["sonyliv_link"]):
                    st.markdown(f"<a class='link' href='{row['sonyliv_link']}' target='_blank'>SonyLiv Link</a>", unsafe_allow_html=True)
                if "yt_link" in row and pd.notna(row["yt_link"]):
                    st.markdown(f"<a class='link' href='{row['yt_link']}' target='_blank'>YouTube Link</a>", unsafe_allow_html=True)
                st.markdown(f"<span class='score'>Score: {row['score']:.3f}</span>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
