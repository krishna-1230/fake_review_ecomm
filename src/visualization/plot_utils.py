"""
Utilities for visualization of fake review detection results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot ROC curve for model evaluation.
    
    Args:
        y_true (array-like): True binary labels
        y_proba (array-like): Predicted probabilities
        model_name (str): Name of the model for the plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'{model_name} ROC Curve',
            line=dict(width=2, color='royalblue')
        )
    )
    
    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Classifier',
            line=dict(width=2, dash='dash', color='gray')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=700,
        height=500
    )
    
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], constrain='domain')
    
    return fig


def plot_precision_recall_curve(y_true, y_proba, model_name="Model"):
    """
    Plot precision-recall curve for model evaluation.
    
    Args:
        y_true (array-like): True binary labels
        y_proba (array-like): Predicted probabilities
        model_name (str): Name of the model for the plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate precision-recall curve points
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Create figure
    fig = go.Figure()
    
    # Add precision-recall curve
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision, mode='lines',
            name=f'{model_name} Precision-Recall Curve',
            line=dict(width=2, color='forestgreen')
        )
    )
    
    # Calculate baseline (no-skill classifier)
    baseline = np.sum(y_true) / len(y_true)
    
    # Add baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[baseline, baseline], mode='lines',
            name='Baseline',
            line=dict(width=2, dash='dash', color='gray')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=700,
        height=500
    )
    
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], constrain='domain')
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true (array-like): True binary labels
        y_pred (array-like): Predicted binary labels
        model_name (str): Name of the model for the plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = ['Real', 'Fake']
    
    # Create heatmap
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect="equal"
    )
    
    # Update layout
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500
    )
    
    return fig


def plot_feature_importances(feature_names, importances, top_n=20, model_name="Model"):
    """
    Plot feature importances from a model.
    
    Args:
        feature_names (list): Names of features
        importances (list): Importance scores
        top_n (int): Number of top features to display
        model_name (str): Name of the model for the plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create DataFrame of features and importances
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and select top_n
    df = df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        title=f'Top {top_n} Feature Importances - {model_name}',
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        width=800,
        height=600
    )
    
    return fig


def plot_review_graph(G, highlight_nodes=None):
    """
    Plot a network graph of users, products, and reviews.
    
    Args:
        G (nx.Graph): NetworkX graph with user-product-review relationships
        highlight_nodes (list, optional): List of nodes to highlight (suspected fake reviewers)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create positions for nodes using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Create node traces
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'user']
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
    
    # Extract positions
    user_pos = {n: pos[n] for n in user_nodes}
    product_pos = {n: pos[n] for n in product_nodes}
    
    # Create highlight set
    if highlight_nodes is None:
        highlight_nodes = set()
    else:
        highlight_nodes = set(highlight_nodes)
    
    # Create traces for users
    user_trace = go.Scatter(
        x=[pos[n][0] for n in user_nodes],
        y=[pos[n][1] for n in user_nodes],
        mode='markers',
        marker=dict(
            size=10,
            color=['red' if n in highlight_nodes else 'blue' for n in user_nodes],
            line=dict(width=1, color='black')
        ),
        text=[f"User: {n}" for n in user_nodes],
        hoverinfo='text',
        name='Users'
    )
    
    # Create traces for products
    product_trace = go.Scatter(
        x=[pos[n][0] for n in product_nodes],
        y=[pos[n][1] for n in product_nodes],
        mode='markers',
        marker=dict(
            size=15,
            color='green',
            symbol='square',
            line=dict(width=1, color='black')
        ),
        text=[f"Product: {n}" for n in product_nodes],
        hoverinfo='text',
        name='Products'
    )
    
    # Create traces for edges
    edge_x = []
    edge_y = []
    edge_text = []
    
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Extract edge data
        rating = d.get('rating', 'N/A')
        date = d.get('date', 'N/A')
        text = f"User: {u}<br>Product: {v}<br>Rating: {rating}<br>Date: {date}"
        edge_text.extend([text, text, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Reviews'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, user_trace, product_trace])
    
    # Update layout
    fig.update_layout(
        title='User-Product Review Network',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=800
    )
    
    return fig


def plot_rating_distribution(reviews_df, by_verified=True, title="Rating Distribution"):
    """
    Plot rating distribution, optionally comparing verified vs. unverified purchases.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        by_verified (bool): Whether to split by verified purchase status
        title (str): Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    if by_verified and 'verified_purchase' in reviews_df.columns:
        # Plot verified purchases
        verified_counts = reviews_df[reviews_df['verified_purchase']]['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=verified_counts.index,
                y=verified_counts.values,
                name='Verified Purchases',
                marker_color='green'
            )
        )
        
        # Plot unverified purchases
        unverified_counts = reviews_df[~reviews_df['verified_purchase']]['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=unverified_counts.index,
                y=unverified_counts.values,
                name='Unverified Purchases',
                marker_color='red'
            )
        )
    else:
        # Plot all ratings
        rating_counts = reviews_df['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color='blue'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Rating',
        yaxis_title='Count',
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5]
        ),
        width=800,
        height=500,
        barmode='group'
    )
    
    return fig


def plot_word_cloud(reviews_df, text_col='review_text', max_words=100):
    """
    Generate a word cloud from review text.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        text_col (str): Name of the column containing review text
        max_words (int): Maximum number of words in the cloud
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("WordCloud package not installed. Please install it with 'pip install wordcloud'")
        return None
    
    # Combine all text
    all_text = ' '.join(reviews_df[text_col].fillna('').astype(str).tolist())
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(all_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Review Text')
    
    return fig


def plot_burstiness(reviews_df, user_id=None, product_id=None):
    """
    Plot burstiness of reviews over time.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame with review data
        user_id (str, optional): Filter by specific user
        product_id (str, optional): Filter by specific product
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Make a copy to avoid modifying original
    df = reviews_df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by user or product if specified
    if user_id:
        df = df[df['user_id'] == user_id]
        title = f"Review Burstiness for User {user_id}"
    elif product_id:
        df = df[df['product_id'] == product_id]
        title = f"Review Burstiness for Product {product_id}"
    else:
        title = "Overall Review Burstiness"
    
    # Compute daily review counts
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='review_count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # Sort by date
    daily_counts = daily_counts.sort_values('date')
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['review_count'],
            mode='lines+markers',
            name='Daily Reviews',
            line=dict(width=2, color='royalblue'),
            marker=dict(size=8)
        )
    )
    
    # Highlight bursts (more than 3 reviews in a day)
    burst_days = daily_counts[daily_counts['review_count'] > 3]
    if not burst_days.empty:
        fig.add_trace(
            go.Scatter(
                x=burst_days['date'],
                y=burst_days['review_count'],
                mode='markers',
                name='Burst (>3 per day)',
                marker=dict(size=12, color='red')
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Reviews',
        width=900,
        height=500
    )
    
    return fig 