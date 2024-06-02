# Fake News Detection

This repository contains a project for detecting fake news using Natural Language Processing (NLP) and Machine Learning techniques. The project leverages pre-trained word vectors and various machine learning algorithms to classify news articles as real or fake.

## Introduction

Fake news has become a significant issue in today's digital age. This project aims to help users identify fake news articles using machine learning models. The project uses a pre-trained Word2Vec model from Google News and several classifiers to predict whether a given news article is real or fake.

## Features

- Preprocesses text data using SpaCy for tokenization and lemmatization.
- Converts text to vectors using pre-trained Word2Vec embeddings.
- Implements and compares various machine learning models including:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Gradient Boosting
  - Random Forest
  - Deep Neural Network (DNN)
- Provides a user-friendly web interface using Streamlit for news classification.

## Installation

To get started with the project, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/Hitesh2498/Fake-News-Detection-NLP-Project.git
```
**2. Install Required dependencies**
```bash
pip install -r requirements.txt
``` 
**3. Download Spacy language model**
```bash
python -m spacy download en_core_web_lg
```
**4. Download [Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)**

**5. vectorize the Data and save in Data Folder**
```bash
python 1.Dataset_vectorization.ipynb
```
**6. Train the models on vectorized data and save in Model folder**
```bash
python 2.Compare_models.ipynb
```
**7. Run the streamlit App**
```bash
streamlit run app.py
```
