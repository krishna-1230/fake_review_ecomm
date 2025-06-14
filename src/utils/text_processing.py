"""
Text processing utilities for NLP analysis of reviews.
"""
import re
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


def clean_text(text):
    """
    Clean and normalize text by removing special characters, 
    converting to lowercase, etc.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text, remove_stopwords=True):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text to tokenize
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        list: List of tokens
    """
    # Ensure NLTK resources are downloaded
    download_nltk_resources()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    
    return tokens


def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their base form.
    
    Args:
        tokens (list): List of tokens to lemmatize
        
    Returns:
        list: List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def extract_ngrams(tokens, n=2):
    """
    Extract n-grams from tokens.
    
    Args:
        tokens (list): List of tokens
        n (int): n-gram size
        
    Returns:
        list: List of n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def calculate_text_statistics(text):
    """
    Calculate statistics from text for feature engineering.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of text statistics
    """
    # Clean and tokenize
    clean = clean_text(text)
    tokens = tokenize_text(clean, remove_stopwords=False)
    
    # Calculate statistics
    stats = {
        'char_count': len(text),
        'word_count': len(tokens),
        'avg_word_length': sum(len(word) for word in tokens) / max(len(tokens), 1),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        'unique_words_ratio': len(set(tokens)) / max(len(tokens), 1),
    }
    
    return stats


def extract_text_features(text):
    """
    Extract text features for model input.
    
    Args:
        text (str): Input review text
        
    Returns:
        dict: Dictionary of text features
    """
    # Get basic text statistics
    stats = calculate_text_statistics(text)
    
    # Clean and process text
    clean = clean_text(text)
    tokens = tokenize_text(clean)
    lemmas = lemmatize_tokens(tokens)
    
    # Get n-grams
    bigrams = extract_ngrams(lemmas, 2)
    trigrams = extract_ngrams(lemmas, 3)
    
    # Get top n-grams
    top_bigrams = Counter(bigrams).most_common(5)
    top_trigrams = Counter(trigrams).most_common(5)
    
    # Create features
    features = {
        **stats,
        'top_bigrams': top_bigrams,
        'top_trigrams': top_trigrams
    }
    
    return features 