"""
Configuration settings for the RSS summarizer application
"""

# File paths
DEFAULT_RSS_PATH = "rss.xml"
DEFAULT_OUTPUT_PATH = "processed_feed.json"

# Processing settings
MAX_ENTRIES = 5  # Maximum number of entries to process

# Summarization settings
TEXTRANK_SENTENCES = 3  # Number of sentences for TextRank summarization
# Number of sentences for TextRank phase in hybrid summarization
HYBRID_TEXTRANK_SENTENCES = 10

# BART summarization settings
BART_MAX_LENGTH = 100  # Maximum length of the BART summary
BART_MIN_LENGTH = 30   # Minimum length of the BART summary
# Model for CPU-friendly inference
BART_MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

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
