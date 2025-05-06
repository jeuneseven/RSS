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

    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a hybrid summary by first using extractive, then abstractive methods

        Args:
            text: Input text to summarize
            **kwargs: Additional parameters

        Returns:
            Hybrid summary
        """
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."

        try:
            # Step 1: Extract sentences with extractive summarizer
            extractive_sentences = kwargs.get("extractive_sentences", 10)
            extractive_summary = self.extractive_summarizer.summarize(
                text, sentences_count=extractive_sentences)

            # Step 2: Generate abstractive summary from the extracted sentences
            max_length = kwargs.get("max_length", 100)
            min_length = kwargs.get("min_length", 30)

            abstractive_params = {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": kwargs.get("do_sample", False),
                "num_beams": kwargs.get("num_beams", 4),
                "early_stopping": kwargs.get("early_stopping", True)
            }

            final_summary = self.abstractive_summarizer.summarize(
                extractive_summary, **abstractive_params)

            return final_summary

        except Exception as e:
            print(f"Error in hybrid summarization: {e}")
            # Fallback to simple extractive summary
            return self.extractive_summarizer.summarize(text, sentences_count=3)
