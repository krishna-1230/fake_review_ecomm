"""
Module for loading and processing review data from various sources.
"""
import os
import pandas as pd


def load_data(reviews_path, user_metadata_path=None, product_metadata_path=None):
    """
    Load review data and related metadata.
    
    Args:
        reviews_path (str): Path to reviews CSV file
        user_metadata_path (str, optional): Path to user metadata CSV file
        product_metadata_path (str, optional): Path to product metadata CSV file
        
    Returns:
        tuple: Tuple containing DataFrames (reviews_df, users_df, products_df)
    """
    # Load reviews
    reviews_df = pd.read_csv(reviews_path)
    
    # Convert review date to datetime
    reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    
    # Convert verified_purchase to boolean
    if 'verified_purchase' in reviews_df.columns:
        reviews_df['verified_purchase'] = reviews_df['verified_purchase'].astype(bool)
    
    # Load user metadata if available
    users_df = None
    if user_metadata_path and os.path.exists(user_metadata_path):
        users_df = pd.read_csv(user_metadata_path)
        
        # Convert join date to datetime if it exists
        if 'join_date' in users_df.columns:
            users_df['join_date'] = pd.to_datetime(users_df['join_date'])
    
    # Load product metadata if available
    products_df = None
    if product_metadata_path and os.path.exists(product_metadata_path):
        products_df = pd.read_csv(product_metadata_path)
        
        # Convert listing date to datetime if it exists
        if 'listing_date' in products_df.columns:
            products_df['listing_date'] = pd.to_datetime(products_df['listing_date'], errors='coerce')
            # Drop rows with invalid dates
            products_df = products_df.dropna(subset=['listing_date'])
    
    return reviews_df, users_df, products_df


def clean_data(reviews_df):
    """
    Clean and prepare review data for analysis.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df = reviews_df.copy()
    
    # Drop rows with missing values in critical columns
    critical_columns = ['user_id', 'product_id', 'review_text']
    df = df.dropna(subset=critical_columns)
    
    # Remove duplicate reviews (same user, product, and text)
    df = df.drop_duplicates(subset=['user_id', 'product_id', 'review_text'])
    
    # Ensure rating is numeric
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        # Fill missing ratings with median or drop
        if df['rating'].isna().any():
            df = df.dropna(subset=['rating'])
    
    # Ensure date is datetime type
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Sort by date
        df = df.sort_values('date')
    
    return df


def prepare_features(reviews_df, users_df=None, products_df=None):
    """
    Prepare initial features for the fake review detection model.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        users_df (pd.DataFrame, optional): DataFrame with user metadata
        products_df (pd.DataFrame, optional): DataFrame with product metadata
        
    Returns:
        pd.DataFrame: DataFrame with added feature columns
    """
    # Make a copy to avoid modifying the original
    df = reviews_df.copy()
    
    # Add basic text features
    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    
    # Add exclamation mark count as a feature
    df['exclamation_count'] = df['review_text'].str.count('!')
    
    # Calculate uppercase ratio
    df['uppercase_ratio'] = df['review_text'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    # Add time-based features if date is available
    if 'date' in df.columns:
        df['hour_of_day'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Group reviews by date to find burst patterns
        daily_counts = df.groupby([df['date'].dt.date, 'user_id']).size().reset_index(name='daily_review_count')
        
        # Merge back to get daily review count for each user
        df = df.merge(
            daily_counts,
            left_on=['date', 'user_id'],
            right_on=[pd.to_datetime(daily_counts['date']).dt.date, 'user_id'],
            how='left'
        )
    
    # Add user metadata features if available
    if users_df is not None:
        df = df.merge(users_df, on='user_id', how='left')
        
        # Calculate days since user joined
        if 'join_date' in df.columns and 'date' in df.columns:
            df['days_since_joining'] = (df['date'] - df['join_date']).dt.days
            
            # Flag new accounts that immediately post reviews
            df['is_new_account_review'] = df['days_since_joining'] <= 7
    
    # Add product metadata features if available
    if products_df is not None:
        df = df.merge(products_df, on='product_id', how='left')
        
        # Calculate days since product was listed
        if 'listing_date' in df.columns and 'date' in df.columns:
            df['days_since_listing'] = (df['date'] - df['listing_date']).dt.days
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): DataFrame to split
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Training and testing DataFrames (train_df, test_df)
    """
    # If 'verified_purchase' column exists, use it as a proxy for ground truth
    if 'verified_purchase' in df.columns:
        # Stratify by verified_purchase to maintain class distribution
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['verified_purchase']
        )
    else:
        # Without ground truth, just split randomly
        train_indices = df.sample(frac=1-test_size, random_state=random_state).index
        train_df = df.loc[train_indices]
        test_df = df.loc[~df.index.isin(train_indices)]
    
    return train_df, test_df 