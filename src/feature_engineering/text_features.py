"""
Module for extracting text-based features from reviews.
"""
import re
import numpy as np
import pandas as pd
from collections import Counter

from src.utils.text_processing import (
    clean_text, 
    tokenize_text, 
    lemmatize_tokens, 
    extract_ngrams,
    calculate_text_statistics
)


def extract_sentiment_features(df, text_col='review_text'):
    """
    Extract sentiment-related features from review text.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Simple sentiment lexicons
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
        'wonderful', 'best', 'love', 'perfect', 'happy', 'recommend', 'pleased',
        'impressive', 'outstanding', 'superb', 'brilliant', 'delightful'
    ])
    
    negative_words = set([
        'bad', 'poor', 'terrible', 'awful', 'worst', 'horrible',
        'disappointed', 'waste', 'useless', 'hate', 'broken', 'defective',
        'refund', 'return', 'complaint', 'disappointing', 'regret'
    ])
    
    # Extreme sentiment words often found in fake reviews
    exaggerated_words = set([
        'best', 'greatest', 'amazing', 'awesome', 'incredible', 'unbelievable',
        'perfect', 'exceptional', 'outstanding', 'revolutionary', 'mind-blowing',
        'astonishing', 'spectacular', 'phenomenal', 'life-changing', 'miraculous'
    ])
    
    # Function to calculate sentiment features
    def get_sentiment_features(text):
        if not isinstance(text, str):
            return {
                'positive_word_count': 0,
                'negative_word_count': 0,
                'sentiment_score': 0,
                'exaggerated_word_count': 0,
                'exaggeration_ratio': 0
            }
        
        # Clean and tokenize text
        clean = clean_text(text)
        tokens = tokenize_text(clean, remove_stopwords=True)
        
        # Count sentiment words
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        exag_count = sum(1 for word in tokens if word in exaggerated_words)
        
        # Calculate ratios
        token_count = len(tokens) if tokens else 1  # Avoid division by zero
        sentiment_score = (pos_count - neg_count) / token_count
        exaggeration_ratio = exag_count / token_count
        
        return {
            'positive_word_count': pos_count,
            'negative_word_count': neg_count,
            'sentiment_score': sentiment_score,
            'exaggerated_word_count': exag_count,
            'exaggeration_ratio': exaggeration_ratio
        }
    
    # Apply the function to each review
    sentiment_features = result_df[text_col].apply(get_sentiment_features).apply(pd.Series)
    
    # Add the features to the DataFrame
    result_df = pd.concat([result_df, sentiment_features], axis=1)
    
    return result_df


def extract_linguistic_features(df, text_col='review_text'):
    """
    Extract linguistic features from review text.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        
    Returns:
        pd.DataFrame: DataFrame with added linguistic features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Function to calculate linguistic features
    def get_linguistic_features(text):
        if not isinstance(text, str):
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'uppercase_ratio': 0,
                'punctuation_ratio': 0,
                'exclamation_ratio': 0,
                'question_ratio': 0
            }
        
        # Calculate word and sentence lengths
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        # Calculate ratios
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Handle empty text
        if not char_count:
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'uppercase_ratio': 0,
                'punctuation_ratio': 0,
                'exclamation_ratio': 0,
                'question_ratio': 0
            }
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Average sentence length (in words)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Character ratios
        uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count
        punctuation_ratio = sum(1 for c in text if c in '.,;:!?"-()[]{}') / char_count
        exclamation_ratio = text.count('!') / char_count
        question_ratio = text.count('?') / char_count
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'uppercase_ratio': uppercase_ratio,
            'punctuation_ratio': punctuation_ratio,
            'exclamation_ratio': exclamation_ratio,
            'question_ratio': question_ratio
        }
    
    # Apply the function to each review
    linguistic_features = result_df[text_col].apply(get_linguistic_features).apply(pd.Series)
    
    # Add the features to the DataFrame
    result_df = pd.concat([result_df, linguistic_features], axis=1)
    
    return result_df


def extract_textual_similarity_features(df, text_col='review_text', group_col='product_id'):
    """
    Extract text similarity features by comparing reviews within groups.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        group_col (str): Column to group by (e.g., product_id, user_id)
        
    Returns:
        pd.DataFrame: DataFrame with added similarity features
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Initialize similarity columns
    result_df['avg_text_similarity'] = 0.0
    result_df['max_text_similarity'] = 0.0
    result_df['similar_review_count'] = 0
    
    # Process each group individually
    for group_value, group_df in result_df.groupby(group_col):
        # Skip if only one review in the group
        if len(group_df) <= 1:
            continue
        
        # Create TF-IDF vectors for the group
        tfidf = TfidfVectorizer(min_df=1, stop_words='english')
        try:
            tfidf_matrix = tfidf.fit_transform(group_df[text_col].fillna(''))
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # For each review, find similarities with others in the same group
            for i, idx in enumerate(group_df.index):
                # Exclude self-similarity (diagonal)
                other_sims = np.delete(similarities[i], i)
                
                if len(other_sims) > 0:
                    # Calculate average and maximum similarity with other reviews
                    avg_sim = np.mean(other_sims)
                    max_sim = np.max(other_sims)
                    
                    # Count highly similar reviews (similarity >= 0.7)
                    similar_count = np.sum(other_sims >= 0.7)
                    
                    # Update the result DataFrame
                    result_df.at[idx, 'avg_text_similarity'] = avg_sim
                    result_df.at[idx, 'max_text_similarity'] = max_sim
                    result_df.at[idx, 'similar_review_count'] = similar_count
        except:
            # Handle cases where TF-IDF fails (e.g., all empty texts)
            pass
    
    return result_df


def combine_text_features(df, text_col='review_text'):
    """
    Combine all text features into a single feature set.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        
    Returns:
        pd.DataFrame: DataFrame with all text features
    """
    # Extract different types of features
    df = extract_sentiment_features(df, text_col)
    df = extract_linguistic_features(df, text_col)
    
    # Extract text statistics using utility function
    def get_text_stats(text):
        return calculate_text_statistics(text)
    
    # Apply the function to each review and convert to DataFrame
    stats_features = df[text_col].apply(get_text_stats).apply(pd.Series)
    
    # Add the features to the DataFrame, avoiding duplicates
    for col in stats_features.columns:
        if col not in df.columns:
            df[col] = stats_features[col]
    
    # Extract similarity features for product groups
    df = extract_textual_similarity_features(df, text_col, 'product_id')
    
    # Extract similarity features for user groups
    df = extract_textual_similarity_features(df, text_col, 'user_id')
    
    return df 