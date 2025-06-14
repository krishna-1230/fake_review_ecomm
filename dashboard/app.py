"""
Main dashboard application for the fake review detector.
"""
import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from dashboard.components.layout import (
    create_navbar,
    create_footer,
    create_dashboard_layout,
    create_review_analysis_layout,
    create_network_graph_layout,
    create_model_insights_layout
)

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = app.server

# Create the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Create invisible div to trigger callbacks on page load
    html.Div(id='_', style={'display': 'none'}),
    
    # Navbar and content area
    create_navbar(),
    
    # Main content container
    dbc.Container([
        html.Div(id='page-content')
    ], fluid=True),
    
    # Footer
    create_footer()
])


# Callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    """
    Update the page content based on the current URL.
    
    Args:
        pathname (str): Current URL pathname
        
    Returns:
        dash component: The page content
    """
    if pathname == '/review-analysis':
        return create_review_analysis_layout()
    elif pathname == '/network-graph':
        return create_network_graph_layout()
    elif pathname == '/model-insights':
        return create_model_insights_layout()
    else:
        return create_dashboard_layout()


# Register callbacks
from dashboard.components.callbacks import register_callbacks
register_callbacks(app)


if __name__ == '__main__':
    # Create the assets directory if it doesn't exist
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Run the app
    app.run_server(debug=True) 