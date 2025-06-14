"""
Module for extracting behavioral features from reviews.
"""
import pandas as pd
import numpy as np
from collections import defaultdict

from src.utils.behavioral_analysis import (
    calculate_burstiness,
    calculate_user_product_metrics,
    build_review_graph
)


def extract_burstiness_features(df):
    """
    Extract features related to review posting patterns and burstiness.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        
    Returns:
        pd.DataFrame: DataFrame with added burstiness features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure date is datetime type
    if 'date' in result_df.columns:
        result_df['date'] = pd.to_datetime(result_df['date'])
    else:
        # Can't calculate time-based features without date
        return result_df
    
    # Initialize burstiness columns
    result_df['user_burstiness_score'] = 0.0
    result_df['product_burstiness_score'] = 0.0
    result_df['max_user_burst'] = 0
    result_df['max_product_burst'] = 0
    
    # Calculate burstiness for each user
    for user_id in result_df['user_id'].unique():
        user_burst = calculate_burstiness(result_df, user_id=user_id)
        user_rows = result_df['user_id'] == user_id
        result_df.loc[user_rows, 'user_burstiness_score'] = user_burst['burstiness_score']
        result_df.loc[user_rows, 'max_user_burst'] = user_burst['max_burst']
    
    # Calculate burstiness for each product
    for product_id in result_df['product_id'].unique():
        product_burst = calculate_burstiness(result_df, product_id=product_id)
        product_rows = result_df['product_id'] == product_id
        result_df.loc[product_rows, 'product_burstiness_score'] = product_burst['burstiness_score']
        result_df.loc[product_rows, 'max_product_burst'] = product_burst['max_burst']
    
    # Calculate time of day and day of week patterns
    result_df['hour_of_day'] = result_df['date'].dt.hour
    result_df['day_of_week'] = result_df['date'].dt.dayofweek
    
    # Group by user to find patterns
    for user_id, user_df in result_df.groupby('user_id'):
        # Skip if only one review
        if len(user_df) <= 1:
            continue
            
        # Calculate hour of day and day of week entropy
        hour_counts = user_df['hour_of_day'].value_counts(normalize=True)
        hour_entropy = -np.sum(hour_counts * np.log2(hour_counts))
        
        day_counts = user_df['day_of_week'].value_counts(normalize=True)
        day_entropy = -np.sum(day_counts * np.log2(day_counts))
        
        # Lower entropy means more concentrated patterns (suspicious)
        user_rows = result_df['user_id'] == user_id
        result_df.loc[user_rows, 'hour_entropy'] = hour_entropy
        result_df.loc[user_rows, 'day_entropy'] = day_entropy
    
    return result_df


def extract_user_behavior_features(df, user_metadata_df=None):
    """
    Extract features related to user behavior.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        
    Returns:
        pd.DataFrame: DataFrame with added user behavior features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate review count per user
    user_review_counts = result_df['user_id'].value_counts()
    result_df['user_review_count'] = result_df['user_id'].map(user_review_counts)
    
    # Calculate unique products reviewed per user
    user_product_counts = result_df.groupby('user_id')['product_id'].nunique()
    result_df['user_product_count'] = result_df['user_id'].map(user_product_counts)
    
    # Calculate average rating per user
    user_avg_ratings = result_df.groupby('user_id')['rating'].mean()
    result_df['user_avg_rating'] = result_df['user_id'].map(user_avg_ratings)
    
    # Calculate rating variance per user (lower variance is suspicious)
    user_rating_variance = result_df.groupby('user_id')['rating'].var()
    result_df['user_rating_variance'] = result_df['user_id'].map(user_rating_variance)
    
    # Calculate deviation from user's average rating
    result_df['rating_deviation'] = abs(result_df['rating'] - result_df['user_avg_rating'])
    
    # Calculate percentage of extreme ratings (1 or 5)
    user_extreme_ratings = result_df[result_df['rating'].isin([1, 5])].groupby('user_id').size()
    user_extreme_pct = user_extreme_ratings / user_review_counts
    result_df['user_extreme_rating_pct'] = result_df['user_id'].map(user_extreme_pct)
    
    # Add metadata features if available
    if user_metadata_df is not None:
        # Merge user metadata
        result_df = result_df.merge(user_metadata_df, on='user_id', how='left')
        
        if 'join_date' in result_df.columns and 'date' in result_df.columns:
            # Ensure dates are datetime
            result_df['join_date'] = pd.to_datetime(result_df['join_date'])
            result_df['date'] = pd.to_datetime(result_df['date'])
            
            # Calculate account age at review time
            result_df['account_age_days'] = (result_df['date'] - result_df['join_date']).dt.days
            
            # Flag suspicious new accounts (posting reviews within 7 days of joining)
            result_df['is_suspicious_new_account'] = result_df['account_age_days'] <= 7
    
    return result_df


def extract_product_behavior_features(df, product_metadata_df=None):
    """
    Extract features related to product review behavior.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
        
    Returns:
        pd.DataFrame: DataFrame with added product behavior features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate review count per product
    product_review_counts = result_df['product_id'].value_counts()
    result_df['product_review_count'] = result_df['product_id'].map(product_review_counts)
    
    # Calculate unique users per product
    product_user_counts = result_df.groupby('product_id')['user_id'].nunique()
    result_df['product_user_count'] = result_df['product_id'].map(product_user_counts)
    
    # Calculate average rating per product
    product_avg_ratings = result_df.groupby('product_id')['rating'].mean()
    result_df['product_avg_rating'] = result_df['product_id'].map(product_avg_ratings)
    
    # Calculate rating variance per product (lower variance is suspicious)
    product_rating_variance = result_df.groupby('product_id')['rating'].var()
    result_df['product_rating_variance'] = result_df['product_id'].map(product_rating_variance)
    
    # Calculate deviation from product's average rating
    result_df['product_rating_deviation'] = abs(result_df['rating'] - result_df['product_avg_rating'])
    
    # Calculate percentage of high ratings (4-5)
    product_high_ratings = result_df[result_df['rating'] >= 4].groupby('product_id').size()
    product_high_pct = product_high_ratings / product_review_counts
    result_df['product_high_rating_pct'] = result_df['product_id'].map(product_high_pct)
    
    # Add metadata features if available
    if product_metadata_df is not None:
        # Merge product metadata
        result_df = result_df.merge(product_metadata_df, on='product_id', how='left')
        
        if 'listing_date' in result_df.columns and 'date' in result_df.columns:
            # Ensure dates are datetime
            result_df['listing_date'] = pd.to_datetime(result_df['listing_date'])
            result_df['date'] = pd.to_datetime(result_df['date'])
            
            # Calculate product age at review time
            result_df['product_age_days'] = (result_df['date'] - result_df['listing_date']).dt.days
            
            # Flag suspicious early reviews (within 3 days of listing)
            result_df['is_suspicious_early_review'] = result_df['product_age_days'] <= 3
    
    return result_df


def extract_graph_features(df):
    """
    Extract graph-based features from review relationships.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        
    Returns:
        pd.DataFrame: DataFrame with added graph features
    """
    import networkx as nx
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Build the review graph
    G = build_review_graph(result_df)
    
    # Initialize graph feature columns
    result_df['user_degree'] = 0
    result_df['product_degree'] = 0
    result_df['user_clustering'] = 0.0
    result_df['user_neighbors_avg_degree'] = 0.0
    
    # Calculate graph metrics for each user and product
    for _, row in result_df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        
        # User degree (number of products reviewed)
        if user_id in G:
            user_degree = G.degree(user_id)
            result_df.loc[_, 'user_degree'] = user_degree
            
            # Get user neighbors (products)
            user_neighbors = list(G.neighbors(user_id))
            
            # Calculate neighbor degrees
            if user_neighbors:
                neighbor_degrees = [G.degree(n) for n in user_neighbors]
                result_df.loc[_, 'user_neighbors_avg_degree'] = sum(neighbor_degrees) / len(neighbor_degrees)
            
        # Product degree (number of reviewers)
        if product_id in G:
            product_degree = G.degree(product_id)
            result_df.loc[_, 'product_degree'] = product_degree
    
    # Create a projection graph of users who reviewed the same products
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
    
    # Create user similarity graph
    user_similarity_graph = nx.Graph()
    
    # Add users as nodes
    for user in user_nodes:
        user_similarity_graph.add_node(user)
    
    # Add edges between users who reviewed the same product
    for product in product_nodes:
        reviewers = list(G.neighbors(product))
        
        # Add edges between all pairs of reviewers
        for i in range(len(reviewers)):
            for j in range(i + 1, len(reviewers)):
                u1, u2 = reviewers[i], reviewers[j]
                
                # Add edge or increment weight if it exists
                if user_similarity_graph.has_edge(u1, u2):
                    user_similarity_graph[u1][u2]['weight'] += 1
                else:
                    user_similarity_graph.add_edge(u1, u2, weight=1)
    
    # Calculate clustering coefficient for each user
    clustering = nx.clustering(user_similarity_graph)
    
    # Add clustering to the result DataFrame
    for user_id, cluster_coef in clustering.items():
        user_rows = result_df['user_id'] == user_id
        result_df.loc[user_rows, 'user_clustering'] = cluster_coef
    
    return result_df


def combine_behavioral_features(df, user_metadata_df=None, product_metadata_df=None):
    """
    Combine all behavioral features into a single feature set.
    
    Args:
        df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
        
    Returns:
        pd.DataFrame: DataFrame with all behavioral features
    """
    # Extract different types of features
    df = extract_burstiness_features(df)
    df = extract_user_behavior_features(df, user_metadata_df)
    df = extract_product_behavior_features(df, product_metadata_df)
    df = extract_graph_features(df)
    
    # Calculate verification ratio if verified_purchase column exists
    if 'verified_purchase' in df.columns:
        df['verified_purchase'] = df['verified_purchase'].astype(bool)
        
        # Calculate verified purchase ratio per user
        user_verified_counts = df[df['verified_purchase']].groupby('user_id').size()
        user_total_counts = df.groupby('user_id').size()
        user_verified_ratio = user_verified_counts / user_total_counts
        
        # Map back to the DataFrame
        df['user_verified_ratio'] = df['user_id'].map(user_verified_ratio)
    
    return df 