import streamlit as st
import pickle
import spacy
import numpy as np
from gensim.downloader import load

# Load pre-trained word vectors
wv = load("word2vec-google-news-300")

# Load SpaCy model for tokenization and lemmatization
nlp = spacy.load("en_core_web_lg")

# Load the trained classifier
with open('./Model/SVM_Model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Function to convert text to vectors
def text_to_vectors(tokens):
    vectors = [wv[token] for token in tokens if token in wv]
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
    else:
        avg_vector = np.zeros(300)
    return avg_vector

# Streamlit app
st.title("Fake News Detection")

# Text input
user_input = st.text_area("Enter the news text:")

# Predict button
if st.button("Predict"):
    if user_input:
        tokens = preprocess_text(user_input)
        vector = text_to_vectors(tokens)
        prediction = clf.predict([vector])[0]
        
        if prediction == 1:
            st.write("The news is **Real**.")
        else:
            st.write("The news is **Fake**.")
    else:
        st.write("Please enter some text to analyze.")
