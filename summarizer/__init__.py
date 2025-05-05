# summarizer/__init__.py
# Export main summarization functionality
from summarizer.summarizer_factory import SummarizerFactory

# Export base classes
from summarizer.base_summarizer import BaseSummarizer
from summarizer.extractive.base_extractive import BaseExtractiveSummarizer
from summarizer.abstractive.base_abstractive import BaseAbstractiveSummarizer
from summarizer.hybrid.base_hybrid import BaseHybridSummarizer

# Export concrete implementations
from summarizer.extractive.textrank_summarizer import TextRankSummarizer
from summarizer.extractive.lexrank_summarizer import LexRankSummarizer
from summarizer.abstractive.bart_summarizer import BARTSummarizer
from summarizer.abstractive.t5_summarizer import T5Summarizer
from summarizer.hybrid.textrank_bart_summarizer import TextRankBARTSummarizer
from summarizer.hybrid.lexrank_t5_summarizer import LexRankT5Summarizer

# Version information
__version__ = '1.0.0'
