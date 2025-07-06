"""
Main entry point for the fake review detector application.
"""
import os
import argparse
import pandas as pd

from src.data_processing.data_loader import load_data, clean_data
from src.feature_engineering.text_features import combine_text_features
from src.feature_engineering.behavioral_features import combine_behavioral_features
from src.models.ensemble import train_ensemble_model
from src.utils.behavioral_analysis import build_review_graph, identify_suspicious_patterns


def train_model(data_path, user_metadata_path=None, product_metadata_path=None, 
              model_save_path='models/ensemble_model.joblib'):
    """
    Train the fake review detection model.
    
    Args:
        data_path (str): Path to the review data CSV
        user_metadata_path (str, optional): Path to user metadata CSV
        product_metadata_path (str, optional): Path to product metadata CSV
        model_save_path (str): Path to save the trained model
    """
    print("Loading data...")
    reviews_df, users_df, products_df = load_data(
        data_path, user_metadata_path, product_metadata_path
    )
    
    print("Cleaning data...")
    reviews_df = clean_data(reviews_df)
    
    print("Training ensemble model...")
    # Create directory for saving model if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train the model
    model, metrics = train_ensemble_model(
        reviews_df,
        users_df,
        products_df,
        save_path=model_save_path
    )
    
    print("Model training completed.")
    print(f"Model saved to: {model_save_path}")
    
    return model, metrics


def analyze_reviews(model, data_path, user_metadata_path=None, product_metadata_path=None, 
                 output_path='data/processed/analysis_results.csv'):
    """
    Analyze reviews and identify potential fake reviews.
    
    Args:
        model: Trained model
        data_path (str): Path to the review data CSV
        user_metadata_path (str, optional): Path to user metadata CSV
        product_metadata_path (str, optional): Path to product metadata CSV
        output_path (str): Path to save analysis results
    """
    print("Loading data...")
    reviews_df, users_df, products_df = load_data(
        data_path, user_metadata_path, product_metadata_path
    )
    
    print("Cleaning data...")
    reviews_df = clean_data(reviews_df)
    
    # If no model is provided, use behavioral analysis only
    if model is None:
        print("No model provided. Using behavioral analysis only.")
        
        print("Building review graph...")
        G = build_review_graph(reviews_df)
        
        print("Identifying suspicious patterns...")
        suspicious_patterns = identify_suspicious_patterns(G)
        
        # Mark suspicious users and products in the DataFrame
        suspicious_users = set(suspicious_patterns['suspicious_users'])
        suspicious_products = set(suspicious_patterns['suspicious_products'])
        
        reviews_df['is_suspicious_user'] = reviews_df['user_id'].isin(suspicious_users)
        reviews_df['is_suspicious_product'] = reviews_df['product_id'].isin(suspicious_products)
        
        # Mark suspicious if either user or product is suspicious
        reviews_df['is_suspicious'] = (reviews_df['is_suspicious_user'] | 
                                     reviews_df['is_suspicious_product'])
    else:
        print("Analyzing reviews with trained model...")
        
        # Predict probability of being fake
        reviews_df['fake_probability'] = model.predict_proba(reviews_df)
        
        # Mark suspicious if probability exceeds threshold
        reviews_df['is_suspicious'] = reviews_df['fake_probability'] >= 0.7
    
    # Save results
    print("Saving analysis results...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reviews_df.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")
    
    # Print summary
    suspicious_count = reviews_df['is_suspicious'].sum()
    total_count = len(reviews_df)
    print(f"Summary: {suspicious_count} out of {total_count} reviews flagged as potentially fake.")
    
    if suspicious_count > 0:
        print("\nTop suspicious reviews:")
        if 'fake_probability' in reviews_df.columns:
            top_suspicious = reviews_df[reviews_df['is_suspicious']].sort_values(
                'fake_probability', ascending=False
            ).head(5)
        else:
            top_suspicious = reviews_df[reviews_df['is_suspicious']].head(5)
            
        for i, (_, row) in enumerate(top_suspicious.iterrows(), 1):
            print(f"{i}. User: {row['user_id']}, Product: {row['product_id']}, " + 
                 f"Rating: {row['rating']}, Date: {row['date']}")
            print(f"   Text: \"{row['review_text'][:100]}...\"")
            print()
    
    return reviews_df


def launch_dashboard():
    """
    Launch the fake review detection dashboard.
    """
    try:
        # Import here to avoid dependency issues if Dash is not installed
        from dashboard.app import app
        print("Launching dashboard...")
        app.run(debug=True)
    except ImportError as e:
        print("Error: Dash dependencies not found. Please install them with:")
        print("pip install dash dash-bootstrap-components")
        print("Actual ImportError:", e)
    except Exception as e:
        print("An unexpected error occurred while launching the dashboard:")
        print(e)


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Fake Review Detector - NLP + Behavioral Modeling"
    )
    
    # Add command subparsers
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the fake review detection model')
    train_parser.add_argument('--data-path', required=True, help='Path to review data CSV')
    train_parser.add_argument('--user-path', help='Path to user metadata CSV')
    train_parser.add_argument('--product-path', help='Path to product metadata CSV')
    train_parser.add_argument('--save-path', default='models/ensemble_model.joblib',
                            help='Path to save the trained model')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze reviews for fake detection')
    analyze_parser.add_argument('--data-path', required=True, help='Path to review data CSV')
    analyze_parser.add_argument('--user-path', help='Path to user metadata CSV')
    analyze_parser.add_argument('--product-path', help='Path to product metadata CSV')
    analyze_parser.add_argument('--model-path', help='Path to trained model')
    analyze_parser.add_argument('--output-path', default='data/processed/analysis_results.csv',
                              help='Path to save analysis results')
    
    # Dashboard command
    subparsers.add_parser('dashboard', help='Launch the fake review detection dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == 'train':
        train_model(
            args.data_path,
            args.user_path,
            args.product_path,
            args.save_path
        )
    elif args.command == 'analyze':
        model = None
        if args.model_path:
            try:
                from src.models.ensemble import EnsembleDetector
                model = EnsembleDetector(model_path=args.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Proceeding with behavioral analysis only.")
        
        analyze_reviews(
            model,
            args.data_path,
            args.user_path,
            args.product_path,
            args.output_path
        )
    elif args.command == 'dashboard':
        launch_dashboard()
    else:
        # Default to launching dashboard if no command is specified
        print("No command specified. Launching dashboard...")
        launch_dashboard()


if __name__ == '__main__':
    main() 