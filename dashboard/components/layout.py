"""
Layout components for the dashboard interface.
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html


def create_navbar():
    """
    Create a navigation bar for the dashboard.
    
    Returns:
        dash component: A navbar component
    """
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="/assets/logo.png",
                        height="40px",
                        className="me-2"
                    ),
                    dbc.NavbarBrand("Fake Review Detector", className="ms-2"),
                ], width="auto"),
            ], align="center", className="g-0"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Review Analysis", href="/review-analysis", active="exact")),
                        dbc.NavItem(dbc.NavLink("Network Graph", href="/network-graph", active="exact")),
                        dbc.NavItem(dbc.NavLink("Model Insights", href="/model-insights", active="exact")),
                    ], navbar=True)
                ], width="auto"),
            ], align="center"),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4",
    )
    
    return navbar


def create_footer():
    """
    Create a footer for the dashboard.
    
    Returns:
        dash component: A footer component
    """
    footer = html.Footer(
        dbc.Container([
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Fake Review Detector - NLP + Behavioral Modeling", className="text-center text-muted"),
                    html.P("© 2023", className="text-center text-muted"),
                ], width=12)
            ])
        ], fluid=True),
        className="mt-4"
    )
    
    return footer


def create_dashboard_layout():
    """
    Create the main dashboard layout.
    
    Returns:
        dash component: The main dashboard layout
    """
    layout = html.Div([
        # Title row
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard Overview"),
                html.P("Overview of fake review detection metrics and insights")
            ], width=12)
        ], className="mb-4"),
        
        # Summary cards row
        dbc.Row([
            # Total reviews card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Reviews", className="card-title"),
                        html.H3(id="total-reviews-card", children="0"),
                    ])
                ])
            ], width=3),
            
            # Suspected fake reviews card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Suspected Fake Reviews", className="card-title"),
                        html.H3(id="fake-reviews-card", children="0"),
                        html.P(id="fake-reviews-percentage", children="0%")
                    ])
                ], color="danger", outline=True)
            ], width=3),
            
            # Suspicious users card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Suspicious Users", className="card-title"),
                        html.H3(id="suspicious-users-card", children="0"),
                    ])
                ], color="warning", outline=True)
            ], width=3),
            
            # Flagged products card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Flagged Products", className="card-title"),
                        html.H3(id="flagged-products-card", children="0"),
                    ])
                ], color="info", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Graphs row 1
        dbc.Row([
            # Rating distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rating Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="rating-distribution-graph")
                    ])
                ])
            ], width=6),
            
            # Verification status pie chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Review Verification Status"),
                    dbc.CardBody([
                        dcc.Graph(id="verification-status-graph")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Graphs row 2
        dbc.Row([
            # Review burstiness
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Review Posting Patterns"),
                    dbc.CardBody([
                        dcc.Graph(id="review-burstiness-graph")
                    ])
                ])
            ], width=12),
        ], className="mb-4"),
        
        # Analysis section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Suspicious Activity"),
                    dbc.CardBody([
                        html.Div(id="suspicious-activity-table")
                    ])
                ])
            ], width=12),
        ]),
    ])
    
    return layout


def create_review_analysis_layout():
    """
    Create the review analysis page layout.
    
    Returns:
        dash component: The review analysis layout
    """
    layout = html.Div([
        # Title row
        dbc.Row([
            dbc.Col([
                html.H2("Review Analysis"),
                html.P("Analyze individual reviews with NLP and behavioral models")
            ], width=12)
        ], className="mb-4"),
        
        # Review input section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Review Input"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Label("User ID"),
                            dbc.Input(id="user-id-input", placeholder="Enter user ID (optional)", type="text", className="mb-3"),
                            
                            dbc.Label("Product ID"),
                            dbc.Input(id="product-id-input", placeholder="Enter product ID (optional)", type="text", className="mb-3"),
                            
                            dbc.Label("Review Text"),
                            dbc.Textarea(id="review-text-input", placeholder="Enter review text to analyze", className="mb-3", rows=5),
                            
                            dbc.Label("Rating"),
                            dcc.Slider(id="rating-slider", min=1, max=5, step=1, value=5, marks={i: str(i) for i in range(1, 6)}),
                            
                            html.Br(),
                            
                            dbc.Button("Analyze Review", id="analyze-review-button", color="primary", className="mt-3"),
                        ])
                    ])
                ])
            ], width=6),
            
            # Results section
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Results"),
                    dbc.CardBody([
                        html.Div(id="review-analysis-results", children=[
                            html.P("Enter a review and click 'Analyze Review' to see results.")
                        ])
                    ])
                ], className="mb-4"),
                
                # Feature visualization
                dbc.Card([
                    dbc.CardHeader("Feature Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id="feature-visualization-graph")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Text analysis section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Text Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="text-analysis-graph")
                    ])
                ])
            ], width=12),
        ]),
    ])
    
    return layout


def create_network_graph_layout():
    """
    Create the network graph page layout.
    
    Returns:
        dash component: The network graph layout
    """
    layout = html.Div([
        # Title row
        dbc.Row([
            dbc.Col([
                html.H2("Network Graph"),
                html.P("Explore relationships between users, products, and reviews")
            ], width=12)
        ], className="mb-4"),
        
        # Filter section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Graph Filters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Filter Type"),
                                dcc.Dropdown(
                                    id="graph-filter-type",
                                    options=[
                                        {"label": "All Data", "value": "all"},
                                        {"label": "By User", "value": "user"},
                                        {"label": "By Product", "value": "product"},
                                        {"label": "By Suspicious Activity", "value": "suspicious"}
                                    ],
                                    value="all",
                                    clearable=False
                                ),
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Label("Filter Value"),
                                dbc.Input(id="graph-filter-value", placeholder="Enter user/product ID", type="text"),
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Label("Highlight Suspicious"),
                                dbc.Checklist(
                                    id="highlight-suspicious",
                                    options=[{"label": "Highlight suspicious users", "value": True}],
                                    value=[True],
                                    switch=True
                                ),
                            ], width=4),
                        ], className="mb-3"),
                        
                        dbc.Button("Update Graph", id="update-graph-button", color="primary", className="mt-2"),
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Graph section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("User-Product-Review Network"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-graph",
                            type="circle",
                            children=[
                                dcc.Graph(id="network-graph", style={'height': '800px'})
                            ]
                        )
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Stats section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Network Statistics"),
                    dbc.CardBody([
                        html.Div(id="network-stats")
                    ])
                ])
            ], width=12)
        ])
    ])
    
    return layout


def create_model_insights_layout():
    """
    Create the model insights page layout.
    
    Returns:
        dash component: The model insights layout
    """
    layout = html.Div([
        # Title row
        dbc.Row([
            dbc.Col([
                html.H2("Model Insights"),
                html.P("Understanding how the model detects fake reviews")
            ], width=12)
        ], className="mb-4"),
        
        # Model selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Selection"),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="model-selector",
                            options=[
                                {"label": "NLP Model", "value": "nlp"},
                                {"label": "Graph-Based Model", "value": "graph"},
                                {"label": "Ensemble Model", "value": "ensemble"}
                            ],
                            value="ensemble",
                            inline=True
                        ),
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Performance metrics
        dbc.Row([
            # ROC Curve
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ROC Curve"),
                    dbc.CardBody([
                        dcc.Graph(id="roc-curve-graph")
                    ])
                ])
            ], width=6),
            
            # Precision-Recall Curve
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Precision-Recall Curve"),
                    dbc.CardBody([
                        dcc.Graph(id="pr-curve-graph")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Confusion Matrix & Feature Importance
        dbc.Row([
            # Confusion Matrix
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Confusion Matrix"),
                    dbc.CardBody([
                        dcc.Graph(id="confusion-matrix-graph")
                    ])
                ])
            ], width=6),
            
            # Feature Importance
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Importance"),
                    dbc.CardBody([
                        dcc.Graph(id="feature-importance-graph")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Model explanation section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Explanation"),
                    dbc.CardBody([
                        html.Div(id="model-explanation")
                    ])
                ])
            ], width=12)
        ])
    ])
    
    return layout 