# summarizer/extractive/base_extractive.py
from summarizer.base_summarizer import BaseSummarizer
from typing import Dict, Any, List, Optional


class BaseExtractiveSummarizer(BaseSummarizer):
    """
    Base class for all extractive summarizers.
    Extractive summarizers select important sentences from the original text.
    """

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata to indicate extractive type"""
        metadata = super().get_metadata()
        metadata.update({
            "type": "extractive",
            "description": "Base extractive summarizer"
        })
        return metadata

    def get_ranked_sentences(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get sentences ranked by importance

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters

        Returns:
            List of dictionaries with sentence text and score
        """
        # Default implementation returns sentences in order with equal weights
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [{"text": s, "score": 1.0} for s in sentences]
