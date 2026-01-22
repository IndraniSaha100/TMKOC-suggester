# TMKOC-Suggester

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)

## Overview
**TMKOC-Suggester** is a **recommendation system for the popular Indian TV show *Taarak Mehta Ka Ooltah Chashmah (TMKOC)***.  
It helps fans discover episodes, jokes, or characters based on their preferences by leveraging **data-driven techniques**, improving engagement and making content exploration fun and personalized.

---

## Features
- Personalized episode or scene recommendations based on user input.
- Character-focused suggestions for fans of specific cast members.
- Episode similarity analysis using **content-based filtering**.
- Optional tracking of user interactions to improve recommendations over time.

---

## Technologies Used
- **Python** – Core programming language.
- **Pandas & NumPy** – Data processing and manipulation.
- **Scikit-learn** – Machine learning models and similarity calculations.
- **NLP Tools** – TF-IDF and Word2Vec for processing episode descriptions or dialogues.
- **Streamlit** – Interactive web interface for recommendations.

---

## Dataset
- **Source:** [TMKOC Episode Dataset on Kaggle](https://www.kaggle.com/datasets/rishabhbhartiya/taarak-mehta-ka-ooltah-chashmah-episode-dateset)
- **Contains:**
  - Episode titles
  - Episode summaries

---

## System Architecture

1. **Data Collection & Preprocessing**
   - Cleaning and structuring episode data.
   - Text preprocessing (removing stop words, tokenization, etc.).
2. **Feature Extraction**
   - Converting episode summaries into numerical vectors (TF-IDF, Word2Vec).
   - Calculating similarity scores between episodes.
3. **Recommendation Engine**
   - Content-based filtering to recommend episodes similar to user preferences.
   - Optionally, collaborative filtering if user ratings become available.
4. **User Interface**
   - Streamlit-based web interface for easy interaction.
   - Input favorite episode, scene, or character to get recommendations.

---

## How to Use

1. **Clone the repository**
   git clone <repository_url>
   cd TMKOC-Suggester
   
2.**Install dependencies**
  pip install -r requirements.txt
  
3.**Run the application**
  streamlit run app.py
  
4.**Interact with the system**
  Enter your favorite episode, character, or scene.
  Get instant recommendations for similar content.
