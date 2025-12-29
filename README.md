# Sentiment-Analysis-using-Naive-Bayes-Text-Classification
# Naïve Bayes Sentiment Analysis – Assignment 1 (EDS 6352)

This project implements a **Naïve Bayes sentiment classifier from scratch** as part of  
**EDS 6352 – Natural Language Processing (Fall 2025)**. The goal is to classify IMDb movie
reviews into positive and negative sentiment using a fully manual probabilistic pipeline
(without scikit‑learn’s Naïve Bayes models).

The notebook includes custom preprocessing, manual sampling, probability estimation,
Laplace smoothing, and evaluation across four preprocessing scenarios.

## Project Overview

- Dataset: IMDb movie reviews (Kaggle)
- Task: Binary sentiment classification (positive vs. negative)
- Model: Custom Naïve Bayes classifier
- Tools: Python, Jupyter Notebook, spaCy, regex
- Restrictions: No use of scikit‑learn’s Naïve Bayes implementations

## Methodology

### **1. Manual Data Sampling**
- Removed missing/NULL values
- Randomly sampled:
  - **1000 reviews** for training  
  - **200 reviews** for testing (balanced: 100 positive, 100 negative)
- Sampling performed without replacement

### **2. Preprocessing Pipeline**
A configurable preprocessing function:preprocess_text(text, lemmatize_words=True, remove_stop_words=True, handle_logical_negation=True)

Handles:
- Cleaning special tokens (`<br>`, `<s>`, etc.)
- Preserving punctuation for spaCy
- Optional lemmatization
- Optional stop‑word removal
- Optional logical negation handling

### **3. Naïve Bayes Implementation**
Implemented fully from scratch:
- Prior probabilities for each class
- Likelihood of each word given each class
- Log‑space posterior probability computation
- Laplace smoothing (+1)
- Removal of unknown test‑set words

### **4. Evaluation**
For each scenario, computed:
- Confusion matrix  
- Per‑class precision  
- Per‑class recall  
- Per‑class F1‑score  

## Preprocessing Scenarios

The full pipeline is run **four times**, comparing the effect of preprocessing:

1. **No lemmatization, no stop‑word removal, no negation handling**
2. **Lemmatization only**
3. **Lemmatization + stop‑word removal + logical negation handling**
