# Fake Review Detector

A sophisticated e-commerce fraud-fighting system that detects suspicious reviews at scale using NLP and behavioral modeling.

## Features

- **NLP Classification**: Analyzes review language patterns, sentiment, and burst characteristics
- **Behavioral Modeling**: Identifies suspicious user patterns using graph-based heuristics
- **Relationship Visualization**: Interactive graphs of user-review-product relationships
- **Spam Gang Detection**: Identifies coordinated fake review networks
- **Dashboard**: Interactive visualization of detection results

## Project Structure

```
fake_review_detector/
├── data/                      # Sample data and processed datasets
├── models/                    # Trained model files
├── src/
│   ├── data_processing/       # Data loading and processing scripts
│   ├── feature_engineering/   # Feature extraction and engineering
│   ├── models/                # ML model implementations
│   │   ├── nlp_model.py       # NLP-based review classification
│   │   ├── graph_model.py     # Graph-based behavioral analysis
│   │   └── ensemble.py        # Combined model approach
│   ├── visualization/         # Visualization components
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for analysis
├── dashboard/                 # Interactive Dash application
│   ├── assets/                # Static assets for the dashboard
│   ├── components/            # Dashboard UI components
│   └── app.py                 # Main dashboard application
├── tests/                     # Unit and integration tests
├── main.py                    # Main application entry point
└── requirements.txt           # Project dependencies
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required NLTK and spaCy resources:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```
   python main.py
   ```

5. Access the dashboard:
   Open your browser and navigate to `http://127.0.0.1:8050/`

## How It Works

1. **Data Processing**: Cleans and preprocesses review data
2. **Feature Engineering**:
   - Text features: sentiment, complexity, n-grams, etc.
   - Behavioral features: posting patterns, time bursts, etc.
   - Graph features: user-product relationships

3. **Detection Models**:
   - NLP model for text classification
   - Graph-based model for relationship analysis
   - Ensemble approach for final classification

4. **Visualization**: Interactive dashboard showing detection results and patterns

## License

MIT "# fake_product_review" 
"# fake_product_review" 
"# fake_review_ecomm" 
