"""
Ensemble model that combines NLP and graph-based models for fake review detection.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.models.nlp_model import ReviewNLPClassifier
from src.models.graph_model import ReviewGraphClassifier
from src.feature_engineering.text_features import combine_text_features
from src.feature_engineering.behavioral_features import combine_behavioral_features


class EnsembleDetector:
    """
    Ensemble model that combines NLP and behavioral models for fake review detection.
    """
    
    def __init__(self, nlp_model=None, graph_model=None, model_path=None):
        """
        Initialize the ensemble detector.
        
        Args:
            nlp_model (ReviewNLPClassifier, optional): Pre-trained NLP model
            graph_model (ReviewGraphClassifier, optional): Pre-trained graph model
            model_path (str, optional): Path to load a saved ensemble model
        """
        self.nlp_model = nlp_model
        self.graph_model = graph_model
        self.weights = {'nlp': 0.5, 'graph': 0.5}  # Default equal weights
        self.trained = False
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def train(self, reviews_df, user_metadata_df=None, product_metadata_df=None, 
             text_col='review_text', target_col='verified_purchase', 
             test_size=0.2, random_state=42, save_path=None):
        """
        Train the ensemble model.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
            product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
            text_col (str): Name of the column containing review text
            target_col (str): Name of the column containing target labels
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            save_path (str, optional): Path to save the trained model
            
        Returns:
            self: Trained model
        """
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
        
        # Train NLP model
        print("Training NLP model...")
        from src.models.nlp_model import train_nlp_model
        nlp_model_path = os.path.join(os.path.dirname(save_path), 'nlp_model.joblib') if save_path else None
        self.nlp_model, _ = train_nlp_model(
            reviews_df,
            text_col=text_col,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            save_path=nlp_model_path
        )
        
        # Train graph model
        print("Training graph model...")
        from src.models.graph_model import train_graph_model
        graph_model_path = os.path.join(os.path.dirname(save_path), 'graph_model.joblib') if save_path else None
        self.graph_model, _ = train_graph_model(
            reviews_df,
            user_metadata_df,
            product_metadata_df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            save_path=graph_model_path
        )
        
        # Set default weights
        self.weights = {'nlp': 0.5, 'graph': 0.5}
        
        # Tune weights if we have both classes
        if len(unique_classes) > 1:
            print("Tuning ensemble weights...")
            self._tune_weights(reviews_df, text_col, target_col)
        else:
            print("Skipping weight tuning due to single class data")
        
        # Mark as trained
        self.trained = True
        
        # Save the model if path is provided
        if save_path:
            self._save_model(save_path)
        
        return self
    
    def predict(self, reviews_df, text_col='review_text'):
        """
        Predict if reviews are fake or real.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            text_col (str): Name of the column containing review text
            
        Returns:
            array: Predicted labels (1 for fake, 0 for real)
        """
        if not self.trained or self.nlp_model is None or self.graph_model is None:
            raise ValueError("Model not fully trained or loaded.")
        
        # Get probabilities from both models
        nlp_proba = self._get_nlp_probabilities(reviews_df, text_col)
        graph_proba = self._get_graph_probabilities(reviews_df)
        
        # Apply weights to probabilities
        ensemble_proba = (self.weights['nlp'] * nlp_proba + 
                         self.weights['graph'] * graph_proba)
        
        # Classify based on threshold of 0.5
        predictions = (ensemble_proba >= 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, reviews_df, text_col='review_text'):
        """
        Predict probability of reviews being fake.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            text_col (str): Name of the column containing review text
            
        Returns:
            array: Predicted probabilities
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get probabilities from both models
        try:
            nlp_proba = self._get_nlp_probabilities(reviews_df, text_col)
        except Exception as e:
            print(f"Error getting NLP probabilities: {e}")
            nlp_proba = np.zeros(len(reviews_df))
            
        try:
            graph_proba = self._get_graph_probabilities(reviews_df)
        except Exception as e:
            print(f"Error getting graph probabilities: {e}")
            graph_proba = np.zeros(len(reviews_df))
        
        # Combine probabilities using weights
        ensemble_proba = (
            self.weights['nlp'] * nlp_proba + 
            self.weights['graph'] * graph_proba
        )
        
        return ensemble_proba
    
    def evaluate(self, reviews_df, text_col='review_text', target_col='verified_purchase'):
        """
        Evaluate the model on review data.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            text_col (str): Name of the column containing review text
            target_col (str): Name of the column containing target labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if target_col not in reviews_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
        # For simplicity, we assume verified_purchase is our ground truth
        # Not verified = potentially fake review (1), Verified = real review (0)
        y_true = (~reviews_df[target_col]).astype(int)
        
        # Check if we have both classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class found in evaluation data: {unique_classes}")
            print("Evaluation metrics may not be meaningful.")
        
        # Make predictions
        y_pred = self.predict(reviews_df, text_col)
        
        # Calculate basic metrics
        metrics = {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Print metrics
        print("Ensemble Model Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Calculate ROC AUC if possible (requires two classes)
        if len(unique_classes) > 1:
            try:
                y_proba = self.predict_proba(reviews_df, text_col)
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                print(f"Ensemble Model ROC AUC Score: {metrics['roc_auc']:.4f}")
                
                # Evaluate individual models
                nlp_proba = self._get_nlp_probabilities(reviews_df, text_col)
                nlp_auc = roc_auc_score(y_true, nlp_proba)
                
                graph_proba = self._get_graph_probabilities(reviews_df)
                graph_auc = roc_auc_score(y_true, graph_proba)
                
                print(f"NLP Model ROC AUC Score: {nlp_auc:.4f}")
                print(f"Graph Model ROC AUC Score: {graph_auc:.4f}")
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None
        else:
            print("Skipping ROC AUC calculation due to single class data")
            metrics['roc_auc'] = None
        
        return metrics
    
    def _get_nlp_probabilities(self, reviews_df, text_col='review_text'):
        """
        Get probabilities from NLP model.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            text_col (str): Name of the column containing review text
            
        Returns:
            array: Predicted probabilities from NLP model
        """
        X_text = reviews_df[text_col].fillna('')
        return self.nlp_model.predict_proba(X_text)
    
    def _get_graph_probabilities(self, reviews_df):
        """
        Get probabilities from graph model.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            
        Returns:
            array: Predicted probabilities from graph model
        """
        # Extract graph features
        from src.models.graph_model import extract_graph_based_features
        graph_df = extract_graph_based_features(reviews_df)
        
        # Get feature columns from model
        feature_cols = self.graph_model.feature_columns
        
        # Fill missing values
        for col in feature_cols:
            if col not in graph_df.columns:
                graph_df[col] = 0
            elif graph_df[col].isna().any():
                graph_df[col] = graph_df[col].fillna(0)
        
        # Get predictions
        return self.graph_model.predict_proba(graph_df[feature_cols])
    
    def _tune_weights(self, reviews_df, text_col='review_text', target_col='verified_purchase'):
        """
        Optimize weights for ensemble combination.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame with review data
            text_col (str): Name of the column containing review text
            target_col (str): Name of the column containing target labels
        """
        y_true = (~reviews_df[target_col]).astype(int)
        
        # Get probabilities from both models
        nlp_proba = self._get_nlp_probabilities(reviews_df, text_col)
        graph_proba = self._get_graph_probabilities(reviews_df)
        
        # Try different weights to find the best combination
        best_auc = 0
        best_weights = {'nlp': 0.5, 'graph': 0.5}
        
        for nlp_weight in np.arange(0, 1.1, 0.1):
            graph_weight = 1 - nlp_weight
            
            # Calculate ensemble probabilities
            ensemble_proba = nlp_weight * nlp_proba + graph_weight * graph_proba
            
            # Calculate AUC
            auc = roc_auc_score(y_true, ensemble_proba)
            
            if auc > best_auc:
                best_auc = auc
                best_weights = {'nlp': nlp_weight, 'graph': graph_weight}
        
        print(f"Optimal weights: NLP = {best_weights['nlp']:.2f}, Graph = {best_weights['graph']:.2f}")
        print(f"Ensemble AUC with optimal weights: {best_auc:.4f}")
        
        self.weights = best_weights
    
    def _save_model(self, path):
        """
        Save the trained ensemble model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self.trained or self.nlp_model is None or self.graph_model is None:
            raise ValueError("Model not fully trained or loaded. Cannot save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the ensemble model weights and configurations
        joblib.dump({
            'weights': self.weights
        }, path)
        
        # Save individual models if not already saved
        nlp_path = os.path.join(os.path.dirname(path), 'nlp_model.joblib')
        graph_path = os.path.join(os.path.dirname(path), 'graph_model.joblib')
        
        if not os.path.exists(nlp_path):
            self.nlp_model._save_model(nlp_path)
            
        if not os.path.exists(graph_path):
            self.graph_model._save_model(graph_path)
    
    def _load_model(self, path):
        """
        Load a trained ensemble model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        
        # Load the ensemble configuration
        ensemble_data = joblib.load(path)
        self.weights = ensemble_data['weights']
        
        # Load individual models
        nlp_path = os.path.join(os.path.dirname(path), 'nlp_model.joblib')
        graph_path = os.path.join(os.path.dirname(path), 'graph_model.joblib')
        
        if os.path.exists(nlp_path):
            self.nlp_model = ReviewNLPClassifier(model_path=nlp_path)
        else:
            raise ValueError(f"NLP model file not found: {nlp_path}")
            
        if os.path.exists(graph_path):
            self.graph_model = ReviewGraphClassifier(model_path=graph_path)
        else:
            raise ValueError(f"Graph model file not found: {graph_path}")
        
        self.trained = True


def train_ensemble_model(reviews_df, user_metadata_df=None, product_metadata_df=None,
                       text_col='review_text', target_col='verified_purchase', 
                       test_size=0.2, random_state=42, save_path=None):
    """
    Train an ensemble model for fake review detection.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_metadata_df (pd.DataFrame, optional): DataFrame with user metadata
        product_metadata_df (pd.DataFrame, optional): DataFrame with product metadata
        text_col (str): Name of the column containing review text
        target_col (str): Name of the column containing target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        save_path (str, optional): Path to save the trained model
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    # Create and train the ensemble model
    model = EnsembleDetector()
    model.train(
        reviews_df, user_metadata_df, product_metadata_df,
        text_col=text_col, target_col=target_col,
        test_size=test_size, random_state=random_state,
        save_path=save_path
    )
    
    # Evaluate the model
    metrics = model.evaluate(reviews_df, text_col=text_col, target_col=target_col)
    
    return model, metrics 