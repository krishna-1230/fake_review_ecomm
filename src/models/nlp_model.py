"""
NLP model for detecting fake reviews based on text content.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class ReviewNLPClassifier:
    """
    Classifier for fake review detection based on NLP features.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the NLP classifier.
        
        Args:
            model_path (str, optional): Path to load a saved model
        """
        self.pipeline = None
        self.feature_names = None
        self.trained = False
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _create_pipeline(self, max_features=5000):
        """
        Create the NLP processing pipeline.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        # Create pipeline with TF-IDF and classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
        
        return pipeline
    
    def train(self, X, y, param_grid=None, cv=5, save_path=None):
        """
        Train the NLP model on review text.
        
        Args:
            X (array-like): Review text data
            y (array-like): Target labels (1 for fake, 0 for real)
            param_grid (dict, optional): Parameter grid for grid search
            cv (int): Number of cross-validation folds
            save_path (str, optional): Path to save the trained model
            
        Returns:
            self: Trained model
        """
        # Create the pipeline
        self.pipeline = self._create_pipeline()
        
        # Check if we have both classes in the target
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class found in training data: {unique_classes}")
            print("Using a simple classifier with no grid search")
            # Train with default parameters
            self.pipeline.fit(X, y)
        else:
            # Set up parameter grid for grid search if provided
            if param_grid:
                grid_search = GridSearchCV(
                    self.pipeline,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit the grid search
                grid_search.fit(X, y)
                
                # Get the best pipeline
                self.pipeline = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_:.4f}")
            else:
                # Train the pipeline with default parameters
                self.pipeline.fit(X, y)
        
        # Get feature names from TF-IDF
        self.feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
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
            X (array-like): Review text data
            
        Returns:
            array: Predicted labels (1 for fake, 0 for real)
        """
        if not self.trained or self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probability of reviews being fake.
        
        Args:
            X (array-like): Review text data
            
        Returns:
            array: Predicted probabilities
        """
        if not self.trained or self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get the probabilities
        proba = self.pipeline.predict_proba(X)
        
        # Check the shape to handle single-class case
        if proba.shape[1] == 1:
            # Only one class, return the raw probabilities
            return proba.flatten()
        else:
            # Two classes, return probability of positive class (index 1)
            return proba[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (array-like): Test review text data
            y_test (array-like): Test target labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.trained or self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
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
    
    def get_top_features(self, n=20):
        """
        Get top features (words/phrases) that indicate fake reviews.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top features
        """
        if not self.trained or self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importances from the classifier
        feature_importances = self.pipeline.named_steps['clf'].feature_importances_
        
        # Sort indices by importance
        indices = np.argsort(feature_importances)[::-1][:n]
        
        # Get feature names
        top_features = [(self.feature_names[i], feature_importances[i]) for i in indices]
        
        return top_features
    
    def _save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self.trained or self.pipeline is None:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump({
            'pipeline': self.pipeline,
            'feature_names': self.feature_names
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
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.trained = True


def train_nlp_model(reviews_df, text_col='review_text', target_col='verified_purchase', 
                  test_size=0.2, random_state=42, save_path=None):
    """
    Train an NLP model for fake review detection.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        target_col (str): Name of the column containing target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        save_path (str, optional): Path to save the trained model
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    # Prepare the data
    X = reviews_df[text_col].fillna('')
    
    # For simplicity, we assume verified_purchase is our ground truth
    # Not verified = potentially fake review (1), Verified = real review (0)
    y = (~reviews_df[target_col]).astype(int) if target_col in reviews_df.columns else None
    
    if y is None:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Check if we have both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class found in data: {unique_classes}")
        print("Model may not perform well with only one class.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(unique_classes) > 1 else None
    )
    
    # Parameter grid for grid search
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [None, 10, 20]
    }
    
    # Create and train the model
    model = ReviewNLPClassifier()
    model.train(X_train, y_train, param_grid=param_grid if len(unique_classes) > 1 else None, save_path=save_path)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics 