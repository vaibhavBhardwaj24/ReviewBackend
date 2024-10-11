# Import necessary libraries
import numpy as np
import pandas as pd
import string
from collections import Counter
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  
# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Define custom stopwords in lowercase
custom_stopwords = [
    "pixel", "samsung", "oneplus", "motorola", "moto", "lg", "htc", "huawei", 
    "asus", "nokia", "sony", "zte", "5a", "xperia", "1", "zenfone", "6", 
    "galaxy", "a9", "7.1", "s23", "open", "z", "fold", "5", "sony", "axon", 
    "m", "oneplus", "u11", "zenfone", "4", "a50", "12", "htc", "galaxy", 
    "a5", "galaxy", "note", "2", "z", "fold", "4", "galaxy", "j4", "a73", 
    "7a", "p40", "model", "lg", "one", "m9", "galaxy", "s5", "mini", 
    "galaxy", "core", "prime", "xr20", "xperia", "5", "xperia", "z5", 
    "premium", "zenfone", "5", "galaxy", "s4", "10", "9", "blade", "v9", 
    "galaxy", "s3", "edge+", "one", "axon", "10", "pro", "s24", "g4", 
    "s24", "ultra", "xperia", "10", "5", "brand", "asus", "v20", "galaxy", 
    "s8", "active", "1", "galaxy", "s2", "zenfone", "v60", "thinq", 
    "nokia", "dex", "huawei", "galaxy", "s7", "edge", "galaxy", "note", 
    "8", "razr", "g8", "thinq", "galaxy", "note", "3", "u12+", "g", 
    "motorola", "galaxy", "note", "4", "pixel", "galaxy", "alpha", 
    "galaxy", "note", "10", "lite", "galaxy", "note", "9", "galaxy", 
    "s7", "s23", "ultra", "samsung", "z", "fold", "6", "9", "pro", 
    "fold", "one", "m8", "one", "a9", "zenfone", "2", "galaxy", "s10e", 
    "zenfone", "3", "galaxy", "k", "zoom", "u", "ultra", "zte", "11", 
    "galaxy", "a3", "galaxy", "note", "galaxy", "note", "edge", "6a", 
    "xperia", "xa2", "rog", "phone", "6", "rog", "phone", "ii", "moto", 
    "fold", "n't","''","``","wo","'s","ca","...","s23+","s23u","android",
    "new","anyone","app","apps","update","updates","still","using"
]

# Update stop_words with custom stopwords in lowercase
stop_words.update([word.lower() for word in custom_stopwords])

# Load the dataset
data = pd.read_csv('overAll.csv')

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()

def remove_stopwords_and_lower(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def get_sentiment_scores(text):
    return sia.polarity_scores(text)

# API Endpoint
@app.route('/device', methods=['POST'])
def sentiment_analysis():
    model = request.json.get('model')
    
    if not model:
        return jsonify({"error": "Model parameter is required"}), 400
    
    # Filter data for the given model
    filtered_data = data[data['model'].str.lower() == model.lower()]

    if filtered_data.empty:
        return jsonify({"error": "Model not found"}), 404

    # Perform sentiment analysis and tokenization
    filtered_data['title'] = filtered_data['title'].str.lower()
    filtered_data['tokenize_title'] = filtered_data['title'].apply(remove_stopwords_and_lower)
    filtered_data['sentiment_scores'] = filtered_data['title'].apply(get_sentiment_scores)

    # Split sentiment scores into separate columns
    filtered_data['positive'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['pos'])
    filtered_data['negative'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['neg'])
    filtered_data['neutral'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['neu'])
    filtered_data['compound'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

    # Get sorted word counts
    punctuation = set(string.punctuation)
    all_tokens = [token for tokens in filtered_data['tokenize_title'] for token in tokens]
    filtered_tokens = [word.lower() for word in all_tokens if word.lower() not in stop_words and word not in punctuation]
    word_counts = Counter(filtered_tokens)
    sorted_word_counts = dict(word_counts.most_common())

    # Prepare the response
    response = {
        "model": model,
        "sentiment_analysis": {
            "positive": filtered_data['positive'].mean(),
            "negative": filtered_data['negative'].mean(),
            "neutral": filtered_data['neutral'].mean(),
            "compound": filtered_data['compound'].mean(),
        },
        "word_count_frequency": sorted_word_counts
    }

    return jsonify(response)
# API Endpoint
@app.route('/dev', methods=['POST'])
def sus():
    model = request.json.get('model')
    
    if not model:
        return jsonify({"error": "Model parameter is required"}), 400
    
    # Filter data for the given model
    filtered_data = data[data['model'].str.lower() == model.lower()].copy()  # Make a copy explicitly

    if filtered_data.empty:
        return jsonify({"error": "Model not found"}), 404

    # Perform sentiment analysis and tokenization
    filtered_data.loc[:, 'title'] = filtered_data['title'].str.lower()  # Use .loc to avoid warning
    filtered_data.loc[:, 'tokenize_title'] = filtered_data['title'].apply(remove_stopwords_and_lower)
    filtered_data.loc[:, 'sentiment_scores'] = filtered_data['title'].apply(get_sentiment_scores)

    # Split sentiment scores into separate columns
    filtered_data.loc[:, 'positive'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['pos'])
    filtered_data.loc[:, 'negative'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['neg'])
    filtered_data.loc[:, 'neutral'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['neu'])
    filtered_data.loc[:, 'compound'] = filtered_data['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

    # Get sorted word counts
    punctuation = set(string.punctuation)
    all_tokens = [token for tokens in filtered_data['tokenize_title'] for token in tokens]
    filtered_tokens = [word.lower() for word in all_tokens if word.lower() not in stop_words and word not in punctuation]
    word_counts = Counter(filtered_tokens)
    
    # Get the top 15 most recurring words
    sorted_word_counts = dict(word_counts.most_common(30))
    review_count = filtered_data.shape[0]  
    # Prepare the response
    response = {
        "model": model,
        "reviewCount":review_count,
        "sentiment_analysis": {
            "positive": filtered_data['positive'].mean(),
            "negative": filtered_data['negative'].mean(),
            "neutral": filtered_data['neutral'].mean(),
            "compound": filtered_data['compound'].mean(),
        },
        "top_word_count_frequency": sorted_word_counts 
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
