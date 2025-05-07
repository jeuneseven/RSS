# config.py
"""
Configuration settings for the RSS summarizer application
"""

# File paths
DEFAULT_RSS_PATH = "index.xml"
DEFAULT_OUTPUT_PATH = "processed_feed.json"

# Processing settings
MAX_ENTRIES = 5  # Maximum number of entries to process

# Summarization settings
DEFAULT_EXTRACTIVE_METHOD = "textrank"  # Default extractive method
DEFAULT_ABSTRACTIVE_METHOD = "bart"     # Default abstractive method
DEFAULT_HYBRID_METHOD = "textrank-bart"  # Default hybrid method

EXTRACTIVE_SENTENCES = 3  # Number of sentences for extractive summarization
# Number of sentences for extractive phase in hybrid summarization
HYBRID_EXTRACTIVE_SENTENCES = 10

# BART summarization settings
BART_MAX_LENGTH = 100  # Maximum length of the BART summary
BART_MIN_LENGTH = 30   # Minimum length of the BART summary
# Model for CPU-friendly inference
BART_MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

# T5 summarization settings
T5_MAX_LENGTH = 100    # Maximum length of the T5 summary
T5_MIN_LENGTH = 30     # Minimum length of the T5 summary
T5_MODEL_NAME = "t5-small"  # Model for CPU-friendly inference

# Classification settings
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"  # Default classification model
DEFAULT_CATEGORIES = [  # Default classification categories
    "Technology",
    "Business",
    "Science",
    "Entertainment",
    "Health"
]
# Minimum confidence threshold for classification predictions
CLASSIFICATION_THRESHOLD = 0.3

# Metrics settings
METRICS = ['rouge-1', 'rouge-2', 'rouge-l']  # ROUGE metrics to calculate

# Display settings
SUMMARY_DISPLAY_LENGTH = 200  # Maximum length to display in formatted results
