"""
Graph-based model for detecting fake reviews using behavioral patterns.
"""
import os
import numpy as np
import pandas as pd
import joblib
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.utils.behavioral_analysis import build_review_graph, identify_suspicious_patterns


class ReviewGraphClassifier:
    """
    Classifier for fake review detection based on graph relationships and behavioral patterns.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the graph classifier.
        
        Args:
            model_path (str, optional): Path to load a saved model
        """
        self.model = None
        self.feature_columns = None
        self.trained = False
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def train(self, X, y, param_grid=None, cv=5, save_path=None):
        """
        Train the graph model on behavioral features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array-like): Target labels (1 for fake, 0 for real)
            param_grid (dict, optional): Parameter grid for grid search
            cv (int): Number of cross-validation folds
            save_path (str, optional): Path to save the trained model
            
        Returns:
            self: Trained model
        """
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Create the model
        base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Check if we have both classes in the target
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class found in training data: {unique_classes}")
            print("Using a simple classifier with no grid search")
            # Train with default parameters
            self.model = base_model
            self.model.fit(X, y)
        else:
            # Set up parameter grid for grid search if provided
            if param_grid:
                grid_search = GridSearchCV(
                    base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit the grid search
                grid_search.fit(X, y)
                
                # Get the best model
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_:.4f}")
            else:
                # Train with default parameters
                self.model = base_model
                self.model.fit(X, y)
        
        # Mark as trained
        self.trained = True
        
        # Save the model if path is provided
        if save_path:
            self._save_model(save_path)
        
        return self
    
    def predict(self, X):
        """
        Predict if reviews are fake or real.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            array: Predicted labels (1 for fake, 0 for real)
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find which feature columns are available in the input data
        available_columns = [col for col in self.feature_columns if col in X.columns]
        
        # Check if we have any available columns
        if not available_columns:
            # If no columns are available, return zeros
            return np.zeros(len(X), dtype=int)
        
        # Create a new DataFrame with only the available columns
        X_available = X[available_columns].copy()
        
        # Add missing columns with zero values
        missing_cols = set(self.feature_columns) - set(available_columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} features. Filling with zeros.")
            for col in missing_cols:
                X_available[col] = 0.0
        
        # Ensure columns are in the right order
        X_model = X_available[self.feature_columns]
        
        # Get the predictions
        try:
            return self.model.predict(X_model)
        except Exception as e:
            print(f"Error in predict: {e}")
            # Return zeros as fallback
            return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X):
        """
        Predict probability of reviews being fake.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            array: Predicted probabilities
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find which feature columns are available in the input data
        available_columns = [col for col in self.feature_columns if col in X.columns]
        
        # Check if we have any available columns
        if not available_columns:
            # If no columns are available, return zeros
            return np.zeros(len(X))
        
        # Create a new DataFrame with only the available columns
        X_available = X[available_columns].copy()
        
        # Add missing columns with zero values
        missing_cols = set(self.feature_columns) - set(available_columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} features. Filling with zeros.")
            for col in missing_cols:
                X_available[col] = 0.0
        
        # Ensure columns are in the right order
        X_model = X_available[self.feature_columns]
        
        # Get the probabilities
        try:
            proba = self.model.predict_proba(X_model)
            
            # Check the shape to handle single-class case
            if proba.shape[1] == 1:
                # Only one class, return the raw probabilities
                return proba.flatten()
            else:
                # Two classes, return probability of positive class (index 1)
                return proba[:, 1]
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            # Return zeros as fallback
            return np.zeros(len(X))
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test feature matrix
            y_test (array-like): Test target labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate basic metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Try to calculate ROC AUC if possible (requires two classes)
        try:
            y_proba = self.predict_proba(X_test)
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
        
        # Print metrics
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def get_feature_importances(self, n=20):
        """
        Get most important features for fake review detection.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top features and their importances
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Sort indices by importance
        indices = np.argsort(importances)[::-1][:n]
        
        # Get feature names and importances
        top_features = [(self.feature_columns[i], importances[i]) for i in indices]
        
        return top_features
    
    def _save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, path)
    
    def _load_model(self, path):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        
        # Load the model
        model_data = joblib.load(path)
        
        # Set model attributes
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.trained = True


def extract_graph_based_features(reviews_df, user_metadata_df=None, product_metadata_df=None):
    """
    Extract graph-based features for fake review detection.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
        
    Returns:
        pd.DataFrame: DataFrame with graph-based features
    """
    # Build the review graph
    G = build_review_graph(reviews_df)
    
    # Get suspicious patterns
    suspicious_patterns = identify_suspicious_patterns(G)
    
    # Initialize feature dataframe
    features = []
    
    # Process each review
    for _, row in reviews_df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        
        user_features = {}
        
        # Add review ID information (as reference)
        user_features['review_id'] = f"{user_id}_{product_id}"
        
        # Basic graph metrics
        user_features['user_degree'] = float(G.degree(user_id)) if G.has_node(user_id) else 0
        user_features['product_degree'] = float(G.degree(product_id)) if G.has_node(product_id) else 0
        
        # User network metrics
        if G.has_node(user_id):
            # Clustering coefficient (local density of connections)
            try:
                user_features['user_clustering'] = nx.clustering(G, user_id)
            except:
                user_features['user_clustering'] = 0.0
                
            # Betweenness centrality (how often user appears on shortest paths)
            if 'betweenness' in nx.__dict__:
                user_betweenness = nx.betweenness_centrality(G, k=min(50, len(G)))
                user_features['user_betweenness'] = user_betweenness.get(user_id, 0.0)
            else:
                user_features['user_betweenness'] = 0.0
                
            # Eigenvector centrality (importance based on connections)
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=100)
                user_features['user_eigenvector_centrality'] = eigenvector.get(user_id, 0.0)
            except:
                user_features['user_eigenvector_centrality'] = 0.0
        else:
            user_features['user_clustering'] = 0.0
            user_features['user_betweenness'] = 0.0
            user_features['user_eigenvector_centrality'] = 0.0
        
        # Compute user similarity metrics
        user_reviews = [n for n in G.neighbors(user_id)] if G.has_node(user_id) else []
        other_users = set()
        
        for prod in user_reviews:
            other_users.update([u for u in G.neighbors(prod) if u != user_id and G.nodes[u].get('type') == 'user'])
        
        # Calculate similarity scores with other users
        similarity_scores = []
        for other_user in other_users:
            other_user_reviews = set([n for n in G.neighbors(other_user)])
            common_products = len(set(user_reviews).intersection(other_user_reviews))
            total_products = len(set(user_reviews).union(other_user_reviews))
            
            if total_products > 0:
                similarity = common_products / total_products
                similarity_scores.append(similarity)
        
        # Compute aggregated similarity metrics
        if similarity_scores:
            user_features['avg_user_similarity'] = sum(similarity_scores) / len(similarity_scores)
            user_features['max_user_similarity'] = max(similarity_scores)
            user_features['similar_user_count'] = float(len(similarity_scores))
        else:
            user_features['avg_user_similarity'] = 0.0
            user_features['max_user_similarity'] = 0.0
            user_features['similar_user_count'] = 0.0
        
        # Add suspicious pattern flags
        user_features['is_suspicious_user'] = float(user_id in suspicious_patterns['suspicious_users'])
        user_features['is_suspicious_product'] = float(product_id in suspicious_patterns['suspicious_products'])
        
        # Add temporal burstiness features if date is available
        if 'date' in reviews_df.columns:
            date = row['date']
            
            # Count reviews by the same user on the same day
            same_day_reviews = reviews_df[(reviews_df['user_id'] == user_id) & 
                                       (reviews_df['date'].dt.date == date.date())]
            user_features['same_day_review_count'] = float(len(same_day_reviews))
            
            # Count reviews for the same product on the same day
            product_same_day = reviews_df[(reviews_df['product_id'] == product_id) & 
                                       (reviews_df['date'].dt.date == date.date())]
            user_features['product_same_day_review_count'] = float(len(product_same_day))
        else:
            user_features['same_day_review_count'] = 0.0
            user_features['product_same_day_review_count'] = 0.0
        
        # Add user metadata features
        if user_metadata_df is not None and 'user_id' in user_metadata_df:
            user_meta = user_metadata_df[user_metadata_df['user_id'] == user_id]
            
            if not user_meta.empty:
                if 'review_count' in user_meta.columns:
                    user_features['user_review_count'] = float(user_meta['review_count'].iloc[0])
                    
                if 'avg_rating' in user_meta.columns:
                    user_features['user_avg_rating'] = float(user_meta['avg_rating'].iloc[0])
                    
                if 'verified_purchases' in user_meta.columns and 'review_count' in user_meta.columns:
                    verified = user_meta['verified_purchases'].iloc[0]
                    total = user_meta['review_count'].iloc[0]
                    if total > 0:
                        user_features['user_verified_ratio'] = float(verified / total)
                    else:
                        user_features['user_verified_ratio'] = 0.0
        
        # Add product metadata features
        if product_metadata_df is not None and 'product_id' in product_metadata_df:
            prod_meta = product_metadata_df[product_metadata_df['product_id'] == product_id]
            
            if not prod_meta.empty:
                if 'avg_rating' in prod_meta.columns:
                    user_features['product_avg_rating'] = float(prod_meta['avg_rating'].iloc[0])
                    
                if 'review_count' in prod_meta.columns:
                    user_features['product_review_count'] = float(prod_meta['review_count'].iloc[0])
        
        # Add to features list
        features.append(user_features)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features)
    
    # Add target column if available in reviews_df
    if 'verified_purchase' in reviews_df.columns:
        features_df['verified_purchase'] = reviews_df['verified_purchase'].values
    
    # Drop any non-numeric columns except the target
    non_numeric_cols = [col for col in features_df.columns 
                       if col != 'verified_purchase' and not pd.api.types.is_numeric_dtype(features_df[col])]
    
    # Drop non-numeric columns including 'review_id'
    features_df = features_df.drop(columns=non_numeric_cols)
    
    # Fill missing values with 0
    features_df = features_df.fillna(0)
    
    return features_df


def train_graph_model(reviews_df, user_metadata_df=None, product_metadata_df=None, 
                     target_col='verified_purchase', test_size=0.2, random_state=42, save_path=None):
    """
    Train a graph-based model for fake review detection.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
        target_col (str): Name of the column containing target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        save_path (str, optional): Path to save the trained model
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    # Extract graph-based features
    features_df = extract_graph_based_features(
        reviews_df, user_metadata_df, product_metadata_df
    )
    
    if target_col not in reviews_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # For simplicity, we assume verified_purchase is our ground truth
    # Not verified = potentially fake review (1), Verified = real review (0)
    y = (~reviews_df[target_col]).astype(int)
    
    # Check if we have both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class found in data: {unique_classes}")
        print("Model may not perform well with only one class.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(unique_classes) > 1 else None
    )
    
    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create and train the model
    model = ReviewGraphClassifier()
    model.train(
        X_train, y_train, 
        param_grid=param_grid if len(unique_classes) > 1 else None, 
        save_path=save_path
    )
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics 