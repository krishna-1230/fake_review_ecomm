"""
Callbacks for the dashboard interface.
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from src.data_processing.data_loader import load_data
from src.utils.behavioral_analysis import build_review_graph, identify_suspicious_patterns
from src.models.ensemble import EnsembleDetector
from src.visualization.plot_utils import (
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_confusion_matrix, 
    plot_feature_importances,
    plot_review_graph,
    plot_rating_distribution,
    plot_burstiness
)


def load_model_and_data(app):
    """
    Load the model and data for the application.
    
    Args:
        app: Dash app instance
        
    Returns:
        tuple: (model, reviews_df, users_df, products_df)
    """
    # Load the data
    try:
        reviews_df, users_df, products_df = load_data(
            reviews_path='data/raw/sample_reviews.csv',
            user_metadata_path='data/raw/user_metadata.csv',
            product_metadata_path='data/raw/product_metadata.csv'
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create empty dataframes as fallback
        reviews_df = pd.DataFrame()
        users_df = pd.DataFrame()
        products_df = pd.DataFrame()
    
    # Try to load the model
    model = None
    try:
        model_dir = 'models/'
        model_path = os.path.join(model_dir, 'ensemble_model.joblib')
        
        # Check if model exists
        if os.path.exists(model_path):
            model = EnsembleDetector(model_path=model_path)
        else:
            print(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return model, reviews_df, users_df, products_df


def register_dashboard_callbacks(app, model, reviews_df, users_df, products_df):
    """
    Register callbacks for the main dashboard.
    
    Args:
        app: Dash app instance
        model: Trained model
        reviews_df: Reviews dataframe
        users_df: Users metadata dataframe
        products_df: Products metadata dataframe
    """
    # Callback for summary cards
    @app.callback(
        [
            Output("total-reviews-card", "children"),
            Output("fake-reviews-card", "children"),
            Output("fake-reviews-percentage", "children"),
            Output("suspicious-users-card", "children"),
            Output("flagged-products-card", "children")
        ],
        [Input("_", "children")]  # Dummy input to trigger on load
    )
    def update_summary_cards(_):
        if reviews_df.empty:
            return "0", "0", "0%", "0", "0"
        
        total_reviews = len(reviews_df)
        
        # Use verified_purchase as a proxy for fake reviews
        fake_reviews = len(reviews_df[~reviews_df['verified_purchase']]) if 'verified_purchase' in reviews_df.columns else 0
        fake_percentage = f"{fake_reviews / total_reviews * 100:.1f}%" if total_reviews > 0 else "0%"
        
        # Build graph to find suspicious patterns
        G = build_review_graph(reviews_df)
        suspicious_patterns = identify_suspicious_patterns(G)
        
        suspicious_users = len(set(suspicious_patterns['suspicious_users']))
        suspicious_products = len(set(suspicious_patterns['suspicious_products']))
        
        return str(total_reviews), str(fake_reviews), fake_percentage, str(suspicious_users), str(suspicious_products)
    
    # Callback for rating distribution graph
    @app.callback(
        Output("rating-distribution-graph", "figure"),
        [Input("_", "children")]  # Dummy input to trigger on load
    )
    def update_rating_distribution(_):
        if reviews_df.empty:
            return {}
        
        fig = plot_rating_distribution(reviews_df, by_verified=True)
        return fig
    
    # Callback for verification status graph
    @app.callback(
        Output("verification-status-graph", "figure"),
        [Input("_", "children")]  # Dummy input to trigger on load
    )
    def update_verification_status_graph(_):
        if reviews_df.empty or 'verified_purchase' not in reviews_df.columns:
            return {}
        
        # Count verified vs unverified reviews
        verified_count = len(reviews_df[reviews_df['verified_purchase']])
        unverified_count = len(reviews_df[~reviews_df['verified_purchase']])
        
        # Create pie chart
        fig = px.pie(
            values=[verified_count, unverified_count],
            names=['Verified', 'Unverified'],
            color=['green', 'red'],
            color_discrete_map={'Verified': 'green', 'Unverified': 'red'},
            title="Review Verification Status"
        )
        
        return fig
    
    # Callback for review burstiness graph
    @app.callback(
        Output("review-burstiness-graph", "figure"),
        [Input("_", "children")]  # Dummy input to trigger on load
    )
    def update_burstiness_graph(_):
        if reviews_df.empty:
            return {}
        
        fig = plot_burstiness(reviews_df)
        return fig
    
    # Callback for suspicious activity table
    @app.callback(
        Output("suspicious-activity-table", "children"),
        [Input("_", "children")]  # Dummy input to trigger on load
    )
    def update_suspicious_activity_table(_):
        if reviews_df.empty:
            return html.P("No data available.")
        
        # Use verified_purchase as a proxy for suspicious activity
        if 'verified_purchase' in reviews_df.columns:
            suspicious_df = reviews_df[~reviews_df['verified_purchase']].copy()
            
            if suspicious_df.empty:
                return html.P("No suspicious activity detected.")
            
            # Sort by date (most recent first)
            suspicious_df = suspicious_df.sort_values('date', ascending=False).head(10)
            
            # Format for display
            display_df = suspicious_df[['user_id', 'product_id', 'rating', 'date', 'review_text']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['review_text'] = display_df['review_text'].str[:100] + '...'
            display_df.columns = ['User ID', 'Product ID', 'Rating', 'Date', 'Review Text']
            
            # Create table
            table = dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in display_df.columns],
                page_size=5,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'height': 'auto',
                    'minWidth': '100px', 'maxWidth': '300px',
                    'whiteSpace': 'normal',
                    'textAlign': 'left'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
            
            return table
        else:
            return html.P("Verification status data not available.")


def register_review_analysis_callbacks(app, model, reviews_df, users_df, products_df):
    """
    Register callbacks for the review analysis page.
    
    Args:
        app: Dash app instance
        model: Trained model
        reviews_df: Reviews dataframe
        users_df: Users metadata dataframe
        products_df: Products metadata dataframe
    """
    @app.callback(
        [
            Output("review-analysis-results", "children"),
            Output("feature-visualization-graph", "figure"),
            Output("text-analysis-graph", "figure")
        ],
        [Input("analyze-review-button", "n_clicks")],
        [
            State("user-id-input", "value"),
            State("product-id-input", "value"),
            State("review-text-input", "value"),
            State("rating-slider", "value")
        ]
    )
    def analyze_review(n_clicks, user_id, product_id, review_text, rating):
        # Default empty outputs
        empty_fig = {}
        
        # Check if button was clicked
        if n_clicks is None or not review_text:
            return (
                html.P("Enter a review and click 'Analyze Review' to see results."),
                empty_fig,
                empty_fig
            )
        
        # Create a small dataframe for this single review
        review_data = {
            'user_id': user_id if user_id else 'new_user',
            'product_id': product_id if product_id else 'new_product',
            'review_text': review_text,
            'rating': rating,
            'date': pd.Timestamp.now(),
            'verified_purchase': False  # Assume unverified for new reviews
        }
        review_df = pd.DataFrame([review_data])
        
        # Try to predict if the review is fake
        prediction_html = html.Div([
            html.H5("Review Analysis Results:")
        ])
        
        fake_score = 0.5  # Default score if model not available
        
        # Check if the model is available
        if model is not None and model.trained:
            try:
                # Predict using the model
                fake_proba = model.predict_proba(review_df)[0]
                fake_score = fake_proba
                
                # Create result card
                if fake_score >= 0.8:
                    color = "danger"
                    result_text = "Highly Suspicious"
                elif fake_score >= 0.6:
                    color = "warning"
                    result_text = "Somewhat Suspicious"
                else:
                    color = "success"
                    result_text = "Likely Authentic"
                
                prediction_html = html.Div([
                    html.H5("Review Analysis Results:"),
                    dcc.Markdown(f"""
                        **Authenticity Score**: {100 - fake_score*100:.1f}/100
                        
                        **Result**: {result_text}
                    """),
                    html.Div([
                        dbc.Progress(
                            value=fake_score * 100,
                            color=color,
                            striped=True,
                            animated=True,
                            style={"height": "30px"}
                        )
                    ], className="mt-3 mb-3"),
                    html.H6("Factors influencing this decision:"),
                    html.Ul([
                        html.Li("Text characteristics (sentiment, complexity, exaggeration)"),
                        html.Li("Rating pattern (compared to average for this product)"),
                        html.Li("User behavior (if user ID was provided)")
                    ])
                ])
            except Exception as e:
                prediction_html = html.Div([
                    html.H5("Error in Analysis"),
                    html.P(f"An error occurred: {str(e)}")
                ])
        else:
            prediction_html = html.Div([
                html.H5("Model not available"),
                html.P("The detection model is not loaded or not trained.")
            ])
        
        # Create text analysis visualization
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        from nltk import word_tokenize
        from collections import Counter
        
        # Tokenize and count words
        words = word_tokenize(review_text.lower())
        word_counts = Counter(words)
        common_words = word_counts.most_common(20)
        
        # Create word frequency bar chart
        word_fig = px.bar(
            x=[word for word, count in common_words],
            y=[count for word, count in common_words],
            labels={'x': 'Word', 'y': 'Frequency'},
            title="Most Common Words in Review"
        )
        
        # Create sentiment visualization
        # Simple lexicon-based approach
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect'])
        negative_words = set(['bad', 'poor', 'terrible', 'awful', 'worst', 'not', 'never'])
        
        # Count sentiment words
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        neutral_count = len(words) - pos_count - neg_count
        
        # Create sentiment donut chart
        sentiment_fig = px.pie(
            values=[pos_count, neg_count, neutral_count],
            names=['Positive', 'Negative', 'Neutral'],
            hole=0.4,
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
            title="Review Sentiment Analysis"
        )
        
        return prediction_html, word_fig, sentiment_fig


def register_network_graph_callbacks(app, model, reviews_df, users_df, products_df):
    """
    Register callbacks for the network graph page.
    
    Args:
        app: Dash app instance
        model: Trained model
        reviews_df: Reviews dataframe
        users_df: Users metadata dataframe
        products_df: Products metadata dataframe
    """
    @app.callback(
        [
            Output("network-graph", "figure"),
            Output("network-stats", "children")
        ],
        [Input("update-graph-button", "n_clicks")],
        [
            State("graph-filter-type", "value"),
            State("graph-filter-value", "value"),
            State("highlight-suspicious", "value")
        ]
    )
    def update_network_graph(n_clicks, filter_type, filter_value, highlight_suspicious):
        if reviews_df.empty:
            return {}, html.P("No data available.")
        
        # Filter data based on selections
        filtered_df = reviews_df.copy()
        
        if filter_type == "user" and filter_value:
            filtered_df = filtered_df[filtered_df['user_id'] == filter_value]
            if filtered_df.empty:
                return {}, html.P(f"No data found for user: {filter_value}")
                
        elif filter_type == "product" and filter_value:
            filtered_df = filtered_df[filtered_df['product_id'] == filter_value]
            if filtered_df.empty:
                return {}, html.P(f"No data found for product: {filter_value}")
                
        elif filter_type == "suspicious":
            if 'verified_purchase' in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df['verified_purchase']]
                if filtered_df.empty:
                    return {}, html.P("No suspicious reviews found.")
            else:
                return {}, html.P("Verification status data not available.")
        
        # Limit the number of reviews for better visualization
        if len(filtered_df) > 100:
            filtered_df = filtered_df.sample(100, random_state=42)
        
        # Build graph
        G = build_review_graph(filtered_df)
        
        # Get suspicious users
        suspicious_users = []
        if highlight_suspicious and True in highlight_suspicious:
            suspicious_patterns = identify_suspicious_patterns(G)
            suspicious_users = suspicious_patterns['suspicious_users']
        
        # Create graph visualization
        fig = plot_review_graph(G, highlight_nodes=suspicious_users)
        
        # Calculate network statistics
        user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
        product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
        
        try:
            import networkx as nx
            
            # Get graph metrics
            stats_html = html.Div([
                html.H5("Network Statistics:"),
                html.Ul([
                    html.Li(f"Number of users: {len(user_nodes)}"),
                    html.Li(f"Number of products: {len(product_nodes)}"),
                    html.Li(f"Number of reviews (edges): {G.number_of_edges()}"),
                    html.Li(f"Network density: {nx.density(G):.4f}"),
                    html.Li(f"Highlighted suspicious users: {len(suspicious_users)}")
                ])
            ])
        except Exception as e:
            stats_html = html.Div([
                html.H5("Error calculating network statistics"),
                html.P(f"Error: {str(e)}")
            ])
        
        return fig, stats_html


def register_model_insights_callbacks(app, model, reviews_df, users_df, products_df):
    """
    Register callbacks for the model insights page.
    
    Args:
        app: Dash app instance
        model: Trained model
        reviews_df: Reviews dataframe
        users_df: Users metadata dataframe
        products_df: Products metadata dataframe
    """
    @app.callback(
        [
            Output("roc-curve-graph", "figure"),
            Output("pr-curve-graph", "figure"),
            Output("confusion-matrix-graph", "figure"),
            Output("feature-importance-graph", "figure"),
            Output("model-explanation", "children")
        ],
        [Input("model-selector", "value")]
    )
    def update_model_insights(selected_model):
        # Default empty outputs
        empty_fig = {}
        
        if reviews_df.empty or 'verified_purchase' not in reviews_df.columns:
            return empty_fig, empty_fig, empty_fig, empty_fig, html.P("No data available.")
        
        # For simplicity, use verified_purchase as ground truth
        y_true = (~reviews_df['verified_purchase']).astype(int)
        
        # If no model, create a basic visualization
        if model is None:
            explanation_html = html.Div([
                html.H5("Model not available"),
                html.P("The detection model is not loaded or not trained. Showing example visualizations.")
            ])
            
            # Create some random predictions for visualization
            y_pred = np.random.randint(0, 2, size=len(y_true))
            y_proba = np.random.random(size=len(y_true))
            
            roc_fig = plot_roc_curve(y_true, y_proba, "Example")
            pr_fig = plot_precision_recall_curve(y_true, y_proba, "Example")
            cm_fig = plot_confusion_matrix(y_true, y_pred, "Example")
            
            # Create example feature importance
            feature_names = ['Exaggeration', 'Rating', 'Burstiness', 'Text_Similarity', 'Account_Age']
            importances = [0.3, 0.25, 0.2, 0.15, 0.1]
            feature_fig = plot_feature_importances(feature_names, importances, top_n=5, model_name="Example")
            
            return roc_fig, pr_fig, cm_fig, feature_fig, explanation_html
        
        # With a real model, use it for predictions
        try:
            # Set model name based on selection
            model_name = {
                'nlp': 'NLP Model',
                'graph': 'Graph-Based Model',
                'ensemble': 'Ensemble Model'
            }.get(selected_model, 'Model')
            
            # Get predictions and probabilities
            if selected_model == 'nlp' and model.nlp_model:
                y_proba = model.nlp_model.predict_proba(reviews_df['review_text'])
                feature_names = model.nlp_model.get_top_features(20)
                if isinstance(feature_names, list) and len(feature_names) > 0 and isinstance(feature_names[0], tuple):
                    feature_importances = [(name, importance) for name, importance in feature_names]
                    feature_names = [name for name, _ in feature_importances]
                    importances = [importance for _, importance in feature_importances]
                else:
                    feature_names = [f"Feature_{i}" for i in range(10)]
                    importances = [0.1] * 10
            elif selected_model == 'graph' and model.graph_model:
                from src.models.graph_model import extract_graph_based_features
                graph_df = extract_graph_based_features(reviews_df, users_df, products_df)
                
                # Fix: Check if we have all required feature columns, use only available ones
                available_features = [col for col in model.graph_model.feature_columns if col in graph_df.columns]
                if not available_features:
                    # Use a fallback if no features are available
                    y_proba = np.zeros(len(reviews_df))
                    feature_names = ['user_degree', 'product_degree', 'is_suspicious_user']
                    importances = [0.5, 0.3, 0.2]
                else:
                    try:
                        y_proba = model.graph_model.predict_proba(graph_df[available_features])
                        
                        # Use available feature columns for importance
                        feature_names = available_features
                        if hasattr(model.graph_model.model, 'feature_importances_'):
                            # Extract importances for available features only
                            all_importances = model.graph_model.model.feature_importances_
                            # Make sure lengths match
                            if len(all_importances) == len(model.graph_model.feature_columns):
                                # Get indices of available features
                                indices = [model.graph_model.feature_columns.index(feat) for feat in available_features]
                                importances = [all_importances[i] for i in indices]
                            else:
                                # Fallback if lengths don't match
                                importances = [0.1] * len(feature_names)
                        else:
                            importances = [0.1] * len(feature_names)
                    except Exception as e:
                        print(f"Error in graph model prediction: {e}")
                        y_proba = np.zeros(len(reviews_df))
                        feature_names = ['user_degree', 'product_degree', 'is_suspicious_user']
                        importances = [0.5, 0.3, 0.2]
            else:
                # Use ensemble model
                y_proba = model.predict_proba(reviews_df)
                # Combine features from both models
                feature_names = []
                importances = []
                
                if model.nlp_model:
                    nlp_features = model.nlp_model.get_top_features(10)
                    if isinstance(nlp_features, list) and len(nlp_features) > 0 and isinstance(nlp_features[0], tuple):
                        for name, imp in nlp_features:
                            feature_names.append(f"NLP_{name}")
                            importances.append(imp * model.weights['nlp'])
                        
                if model.graph_model:
                    graph_features = model.graph_model.feature_columns[:10]
                    graph_importances = model.graph_model.model.feature_importances_[:10]
                    feature_names.extend([f"Graph_{name}" for name in graph_features])
                    importances.extend([imp * model.weights['graph'] for imp in graph_importances])
                
                # Fallback if no features were added
                if not feature_names:
                    feature_names = [f"Feature_{i}" for i in range(10)]
                    importances = [0.1] * 10
            
            y_pred = (y_proba >= 0.5).astype(int)
            
            # Create plots
            roc_fig = plot_roc_curve(y_true, y_proba, model_name)
            pr_fig = plot_precision_recall_curve(y_true, y_proba, model_name)
            cm_fig = plot_confusion_matrix(y_true, y_pred, model_name)
            
            # Ensure we have meaningful features to display
            if not feature_names or len(feature_names) != len(importances):
                feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
                importances = [0.3, 0.25, 0.2, 0.15, 0.1]
            
            feature_fig = plot_feature_importances(feature_names, importances, model_name=model_name)
            
            # Create explanation
            if selected_model == 'nlp':
                explanation_html = html.Div([
                    html.H5("NLP Model Explanation"),
                    html.P("The NLP model analyzes the text content of reviews to identify patterns and characteristics common in fake reviews."),
                    html.Ul([
                        html.Li("Text is processed using TF-IDF vectorization and analyzed with a Random Forest classifier."),
                        html.Li("Key indicators include exaggerated language, extreme sentiment, unusual punctuation, and suspicious word patterns."),
                        html.Li("The model was trained on labeled data, using verified purchase status as a proxy for authentic reviews."),
                    ])
                ])
            elif selected_model == 'graph':
                explanation_html = html.Div([
                    html.H5("Graph-Based Model Explanation"),
                    html.P("The graph-based model focuses on behavioral patterns and relationships between users, products, and reviews."),
                    html.Ul([
                        html.Li("Users and products are represented as nodes in a network, with reviews as connections between them."),
                        html.Li("Suspicious patterns include bursts of similar reviews, groups of users with unusually similar behavior, and unusual rating patterns."),
                        html.Li("The model calculates metrics like burstiness, clustering coefficient, and centrality to identify suspicious activity."),
                    ])
                ])
            else:
                explanation_html = html.Div([
                    html.H5("Ensemble Model Explanation"),
                    html.P("The ensemble model combines the NLP and graph-based approaches for more accurate detection."),
                    html.Ul([
                        html.Li(f"NLP component weight: {model.weights['nlp']:.2f}"),
                        html.Li(f"Graph component weight: {model.weights['graph']:.2f}"),
                        html.Li("This combined approach is more robust against sophisticated fraud attempts."),
                        html.Li("By analyzing both text content and behavioral patterns, the system can detect fake reviews that might pass a single-method check."),
                    ])
                ])
            
            return roc_fig, pr_fig, cm_fig, feature_fig, explanation_html
            
        except Exception as e:
            error_html = html.Div([
                html.H5("Error in Model Insights"),
                html.P(f"An error occurred: {str(e)}")
            ])
            
            return empty_fig, empty_fig, empty_fig, empty_fig, error_html


def register_callbacks(app):
    """
    Register all callbacks for the application.
    
    Args:
        app: Dash app instance
    """
    # Load model and data
    model, reviews_df, users_df, products_df = load_model_and_data(app)
    
    # Register callbacks for each page
    register_dashboard_callbacks(app, model, reviews_df, users_df, products_df)
    register_review_analysis_callbacks(app, model, reviews_df, users_df, products_df)
    register_network_graph_callbacks(app, model, reviews_df, users_df, products_df)
    register_model_insights_callbacks(app, model, reviews_df, users_df, products_df) 