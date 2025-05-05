# summarizer/base_summarizer.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSummarizer(ABC):
    """
    Abstract base class for all summarizers.
    All summarizer implementations must inherit from this class and implement the summarize method.
    """

    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a summary of the input text

        Args:
            text: Input text to summarize
            **kwargs: Additional parameters specific to the summarizer

        Returns:
            Generated summary
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the summarizer

        Returns:
            Dictionary with metadata (name, type, description, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "type": "base",
            "description": "Base summarizer class"
        }

    def __str__(self) -> str:
        """String representation of the summarizer"""
        metadata = self.get_metadata()
        return f"{metadata['name']} ({metadata['type']})"
