# summarizer/abstractive/base_abstractive.py
from summarizer.base_summarizer import BaseSummarizer
from typing import Dict, Any, Optional


class BaseAbstractiveSummarizer(BaseSummarizer):
    """
    Base class for all abstractive summarizers.
    Abstractive summarizers generate new text rather than extracting sentences.
    """

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata to indicate abstractive type"""
        metadata = super().get_metadata()
        metadata.update({
            "type": "abstractive",
            "description": "Base abstractive summarizer"
        })
        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying model

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "none",
            "model_size": "unknown",
            "language": "english"
        }
