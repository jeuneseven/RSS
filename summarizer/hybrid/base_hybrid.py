# summarizer/hybrid/base_hybrid.py
from summarizer.base_summarizer import BaseSummarizer
from typing import Dict, Any, Optional


class BaseHybridSummarizer(BaseSummarizer):
    """
    Base class for all hybrid summarizers.
    Hybrid summarizers combine extractive and abstractive techniques.
    """

    def __init__(self, extractive_summarizer=None, abstractive_summarizer=None):
        """
        Initialize with component summarizers

        Args:
            extractive_summarizer: Instance of an extractive summarizer
            abstractive_summarizer: Instance of an abstractive summarizer
        """
        self.extractive_summarizer = extractive_summarizer
        self.abstractive_summarizer = abstractive_summarizer

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata to indicate hybrid type"""
        metadata = super().get_metadata()
        metadata.update({
            "type": "hybrid",
            "description": "Base hybrid summarizer"
        })
        return metadata

    def get_components(self) -> Dict[str, Any]:
        """
        Get information about component summarizers

        Returns:
            Dictionary with component information
        """
        return {
            "extractive": str(self.extractive_summarizer) if self.extractive_summarizer else "None",
            "abstractive": str(self.abstractive_summarizer) if self.abstractive_summarizer else "None"
        }
