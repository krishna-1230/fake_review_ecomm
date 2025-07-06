# Fake Review Detector

A sophisticated e-commerce fraud-fighting system that detects suspicious reviews at scale using Natural Language Processing (NLP) and behavioral modeling techniques.

## Overview

The Fake Review Detector is designed to identify potentially fraudulent reviews in e-commerce platforms by analyzing both the content of reviews and the behavioral patterns of users. It combines text analysis, user behavior analysis, and graph-based relationship analysis to provide a comprehensive detection system.

## Features

- **NLP Classification**: Analyzes review language patterns, sentiment, and linguistic characteristics
- **Behavioral Modeling**: Identifies suspicious user patterns using time-based and activity-based features
- **Graph Analysis**: Examines user-product relationships to detect coordinated fake review networks
- **Ensemble Approach**: Combines multiple detection methods for improved accuracy
- **Interactive Dashboard**: Visualizes detection results and provides insights into suspicious patterns

## Project Structure

```
fakeReview/
├── dashboard/                # Interactive Dash application
│   ├── app.py               # Main dashboard application
│   ├── assets/              # Static assets for the dashboard
│   └── components/          # Dashboard UI components
├── data/                    # Data directory
│   ├── processed/           # Processed datasets and analysis results
│   └── raw/                 # Raw input data files
├── models/                  # Trained model files
├── src/                     # Source code
│   ├── data_processing/     # Data loading and processing scripts
│   ├── feature_engineering/ # Feature extraction and engineering
│   │   ├── behavioral_features.py # Behavioral feature extraction
│   │   └── text_features.py # Text-based feature extraction
│   ├── models/              # ML model implementations
│   │   ├── ensemble.py      # Combined model approach
│   │   ├── graph_model.py   # Graph-based behavioral analysis
│   │   └── nlp_model.py     # NLP-based review classification
│   ├── utils/               # Utility functions
│   └── visualization/       # Visualization components
├── main.py                  # Main application entry point
├── requirements.txt         # Project dependencies
├── run_fake_review.bat      # Windows batch file to run the application
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/krishna-1230/fake_review_ecomm.git
   cd fakeReview
   ```

2. **Create and activate a virtual environment**:
   ```
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Download required NLTK and spaCy resources**:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_sm
   ```

5. **Run the application**:
   ```
   # On Windows
   run_fake_review.bat
   
   # Or directly with Python
   python main.py dashboard
   ```

6. **Access the dashboard**:
   Open your browser and navigate to `http://127.0.0.1:8050/`

## Usage

The application supports three main modes of operation:

### Training Mode

Train the fake review detection model using your data:

```
python main.py train --data-path data/raw/sample_reviews.csv --user-path data/raw/user_metadata.csv --product-path data/raw/product_metadata.csv
```

### Analysis Mode

Analyze reviews to identify potential fake reviews:

```
python main.py analyze --data-path data/raw/sample_reviews.csv --user-path data/raw/user_metadata.csv --product-path data/raw/product_metadata.csv
```

### Dashboard Mode

Launch the interactive dashboard for visualization and exploration:

```
python main.py dashboard
```

## How It Works

### 1. Data Processing

- Loads and cleans review data, user metadata, and product metadata
- Handles missing values, duplicates, and data type conversions
- Prepares data for feature extraction

### 2. Feature Engineering

- **Text Features**: Extracts linguistic patterns, sentiment, complexity metrics, and n-grams
- **Behavioral Features**: Analyzes posting patterns, time bursts, and user activity metrics
- **Graph Features**: Builds user-product relationship networks to identify suspicious patterns

### 3. Detection Models

- **NLP Model**: Uses natural language processing to classify review text
- **Graph Model**: Analyzes user-product relationships to identify suspicious patterns
- **Ensemble Approach**: Combines both models with optimized weights for final classification

### 4. Visualization

- Interactive dashboard with multiple views:
  - Overview statistics and metrics
  - Detailed review analysis
  - Network graphs of user-product relationships
  - Model insights and performance metrics

## Requirements

- Python 3.8+
- See requirements.txt for complete list of dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on the GitHub repository.
