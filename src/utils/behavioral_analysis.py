"""
Utilities for behavioral analysis and graph-based metrics.
"""
import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx


def calculate_burstiness(reviews_df, user_id=None, product_id=None, window_days=7):
    """
    Calculate review burstiness - how many reviews were posted in quick succession.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_id (str, optional): Filter by specific user
        product_id (str, optional): Filter by specific product
        window_days (int): Time window in days to consider for bursts
        
    Returns:
        dict: Burstiness metrics
    """
    # Filter data if needed
    df = reviews_df.copy()
    if user_id:
        df = df[df['user_id'] == user_id]
    if product_id:
        df = df[df['product_id'] == product_id]
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < 2:
        return {
            'max_burst': 0,
            'burst_intervals': [],
            'burstiness_score': 0.0
        }
    
    # Calculate time differences between consecutive reviews
    df['time_diff'] = df['date'].diff().dt.total_seconds() / (24 * 3600)  # in days
    
    # Identify bursts (reviews within window)
    bursts = df[df['time_diff'] <= window_days]['time_diff'].tolist()
    
    # Calculate burst metrics
    if not bursts:
        max_burst = 0
        burst_intervals = []
        burstiness_score = 0.0
    else:
        max_burst = len(bursts)
        burst_intervals = bursts
        # Burstiness score: ratio of reviews in bursts to total reviews
        burstiness_score = max_burst / len(df)
    
    return {
        'max_burst': max_burst,
        'burst_intervals': burst_intervals,
        'burstiness_score': burstiness_score
    }


def calculate_user_product_metrics(reviews_df, user_metadata_df=None):
    """
    Calculate metrics about user-product relationships.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        
    Returns:
        dict: Dictionary of user-product metrics
    """
    metrics = {}
    
    # User review patterns
    user_review_counts = reviews_df['user_id'].value_counts()
    metrics['users_with_single_review'] = sum(user_review_counts == 1)
    metrics['users_with_multiple_reviews'] = sum(user_review_counts > 1)
    
    # Product review patterns
    product_review_counts = reviews_df['product_id'].value_counts()
    metrics['products_with_single_review'] = sum(product_review_counts == 1)
    metrics['products_with_multiple_reviews'] = sum(product_review_counts > 1)
    
    # User-Product combinations
    user_product_combos = reviews_df.groupby(['user_id', 'product_id']).size()
    metrics['user_product_pairs'] = len(user_product_combos)
    metrics['user_product_pairs_with_multiple_reviews'] = sum(user_product_combos > 1)
    
    # Add user metadata if available
    if user_metadata_df is not None:
        # Join with user metadata
        merged = reviews_df.merge(user_metadata_df, on='user_id', how='left')
        
        # New user percentage (users who joined recently and immediately posted reviews)
        today = pd.to_datetime('today')
        merged['join_date'] = pd.to_datetime(merged['join_date'])
        merged['days_since_joining'] = (merged['date'] - merged['join_date']).dt.days
        new_users = merged[merged['days_since_joining'] <= 7]
        metrics['new_user_review_percentage'] = len(new_users) / len(merged) if len(merged) > 0 else 0
        
        # Verified purchase percentage
        metrics['verified_purchase_percentage'] = merged['verified_purchase'].mean() if 'verified_purchase' in merged.columns else None
    
    return metrics


def build_review_graph(reviews_df):
    """
    Build a graph representing relationships between users, products, and reviews.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        
    Returns:
        nx.Graph: NetworkX graph with user-product-review relationships
    """
    G = nx.Graph()
    
    # Add user nodes
    for user_id in reviews_df['user_id'].unique():
        G.add_node(user_id, type='user')
    
    # Add product nodes
    for product_id in reviews_df['product_id'].unique():
        G.add_node(product_id, type='product')
    
    # Add edges between users and products (representing reviews)
    for _, row in reviews_df.iterrows():
        G.add_edge(row['user_id'], row['product_id'], 
                   rating=row['rating'],
                   date=row['date'],
                   review_id=_ if 'review_id' not in row else row['review_id'])
    
    return G


def identify_suspicious_patterns(G, min_similarity=0.7):
    """
    Identify suspicious patterns in the review graph.
    
    Args:
        G (nx.Graph): NetworkX graph with user-product-review relationships
        min_similarity (float): Minimum similarity threshold for suspicious patterns
        
    Returns:
        dict: Dictionary of suspicious patterns
    """
    suspicious = {
        'suspicious_users': [],
        'suspicious_products': [],
        'suspicious_user_groups': [],
        'suspicious_product_groups': []
    }
    
    # Get user and product nodes
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
    
    # Calculate user similarity based on reviewed products and ratings
    user_similarity = {}
    for u1 in user_nodes:
        u1_products = set(G.neighbors(u1))
        if not u1_products:
            continue
            
        for u2 in user_nodes:
            if u1 >= u2:  # Avoid duplicates
                continue
                
            u2_products = set(G.neighbors(u2))
            if not u2_products:
                continue
                
            # Calculate Jaccard similarity of products reviewed
            intersection = u1_products.intersection(u2_products)
            union = u1_products.union(u2_products)
            jaccard = len(intersection) / len(union) if union else 0
            
            # Calculate rating similarity for common products
            rating_sim = 0
            if intersection:
                rating_diffs = []
                for p in intersection:
                    r1 = G[u1][p].get('rating', 0)
                    r2 = G[u2][p].get('rating', 0)
                    diff = abs(r1 - r2) / 5.0  # Normalized by max rating
                    rating_diffs.append(1.0 - diff)  # Convert to similarity
                
                rating_sim = sum(rating_diffs) / len(rating_diffs) if rating_diffs else 0
            
            # Combined similarity
            combined_sim = (jaccard + rating_sim) / 2
            
            if combined_sim >= min_similarity and len(intersection) > 0:
                pair_key = tuple(sorted([u1, u2]))
                user_similarity[pair_key] = combined_sim
    
    # Form user groups based on similarity
    if user_similarity:
        # Create a graph of similar users
        SG = nx.Graph()
        for (u1, u2), sim in user_similarity.items():
            SG.add_edge(u1, u2, weight=sim)
        
        # Find connected components (potential fake review groups)
        suspicious_groups = list(nx.connected_components(SG))
        suspicious['suspicious_user_groups'] = [list(g) for g in suspicious_groups if len(g) >= 2]
        
        # Add individual suspicious users
        all_suspicious_users = set()
        for group in suspicious['suspicious_user_groups']:
            all_suspicious_users.update(group)
        suspicious['suspicious_users'] = list(all_suspicious_users)
    
    # Find products with unusual patterns
    product_metrics = {}
    for p in product_nodes:
        reviewers = list(G.neighbors(p))
        if not reviewers:
            continue
            
        # Extract ratings
        ratings = [G[p][u].get('rating', 0) for u in reviewers]
        
        # Calculate metrics
        metrics = {
            'review_count': len(reviewers),
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_variance': np.var(ratings) if len(ratings) > 1 else 0,
            'high_rating_pct': sum(1 for r in ratings if r >= 4) / len(ratings) if ratings else 0
        }
        
        # Flag suspicious products (high rating, low variance)
        if (metrics['review_count'] >= 5 and 
            metrics['avg_rating'] >= 4.5 and 
            metrics['rating_variance'] < 0.5 and
            metrics['high_rating_pct'] >= 0.8):
            suspicious['suspicious_products'].append(p)
            
        product_metrics[p] = metrics
    
    # Find groups of products with similar reviewers
    product_similarity = {}
    for p1 in product_nodes:
        p1_users = set(G.neighbors(p1))
        if not p1_users:
            continue
            
        for p2 in product_nodes:
            if p1 >= p2:  # Avoid duplicates
                continue
                
            p2_users = set(G.neighbors(p2))
            if not p2_users:
                continue
                
            # Calculate Jaccard similarity of reviewers
            intersection = p1_users.intersection(p2_users)
            union = p1_users.union(p2_users)
            jaccard = len(intersection) / len(union) if union else 0
            
            if jaccard >= min_similarity and len(intersection) >= 3:
                pair_key = tuple(sorted([p1, p2]))
                product_similarity[pair_key] = jaccard
    
    # Form product groups based on similarity
    if product_similarity:
        # Create a graph of similar products
        PG = nx.Graph()
        for (p1, p2), sim in product_similarity.items():
            PG.add_edge(p1, p2, weight=sim)
        
        # Find connected components (products with similar reviewer sets)
        product_groups = list(nx.connected_components(PG))
        suspicious['suspicious_product_groups'] = [list(g) for g in product_groups if len(g) >= 2]
    
    return suspicious 