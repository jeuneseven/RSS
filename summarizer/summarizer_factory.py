# summarizer/summarizer_factory.py
from typing import Dict, Any, Optional, List


class SummarizerFactory:
    """
    Factory class to create and manage summarizer instances.
    Provides a centralized way to instantiate different summarizers.
    """

    @staticmethod
    def create_extractive_summarizer(name: str, **kwargs) -> Any:
        """
        Create an extractive summarizer instance

        Args:
            name: Name of the summarizer (e.g., 'textrank', 'lexrank')
            **kwargs: Additional parameters for initialization

        Returns:
            Instance of the requested summarizer
        """
        name = name.lower()

        if name == 'textrank':
            from summarizer.extractive.textrank_summarizer import TextRankSummarizer
            return TextRankSummarizer(**kwargs)

        elif name == 'lexrank':
            from summarizer.extractive.lexrank_summarizer import LexRankSummarizer
            return LexRankSummarizer(**kwargs)

        else:
            raise ValueError(f"Unknown extractive summarizer: {name}")

    @staticmethod
    def create_abstractive_summarizer(name: str, **kwargs) -> Any:
        """
        Create an abstractive summarizer instance

        Args:
            name: Name of the summarizer (e.g., 'bart', 't5')
            **kwargs: Additional parameters for initialization

        Returns:
            Instance of the requested summarizer
        """
        name = name.lower()

        if name == 'bart':
            from summarizer.abstractive.bart_summarizer import BARTSummarizer
            return BARTSummarizer(**kwargs)

        elif name == 't5':
            from summarizer.abstractive.t5_summarizer import T5Summarizer
            return T5Summarizer(**kwargs)

        else:
            raise ValueError(f"Unknown abstractive summarizer: {name}")

    @staticmethod
    def create_hybrid_summarizer(name: str, **kwargs) -> Any:
        """
        Create a hybrid summarizer instance

        Args:
            name: Name of the summarizer (e.g., 'textrank-bart', 'lexrank-t5')
            **kwargs: Additional parameters for initialization

        Returns:
            Instance of the requested hybrid summarizer
        """
        name = name.lower()

        if name == 'textrank-bart':
            from summarizer.hybrid.textrank_bart_summarizer import TextRankBARTSummarizer
            return TextRankBARTSummarizer(**kwargs)

        elif name == 'lexrank-t5':
            from summarizer.hybrid.lexrank_t5_summarizer import LexRankT5Summarizer
            return LexRankT5Summarizer(**kwargs)

        else:
            # Try to dynamically create a hybrid from components
            parts = name.split('-')
            if len(parts) == 2:
                try:
                    extractive = SummarizerFactory.create_extractive_summarizer(
                        parts[0])
                    abstractive = SummarizerFactory.create_abstractive_summarizer(
                        parts[1])

                    from summarizer.hybrid.base_hybrid import BaseHybridSummarizer
                    return BaseHybridSummarizer(extractive, abstractive)

                except Exception as e:
                    print(f"Error creating dynamic hybrid: {e}")

            raise ValueError(f"Unknown hybrid summarizer: {name}")

    @staticmethod
    def get_available_summarizers() -> Dict[str, List[str]]:
        """
        Get a list of all available summarizers

        Returns:
            Dictionary with summarizer types and names
        """
        return {
            "extractive": ["textrank", "lexrank"],
            "abstractive": ["bart", "t5"],
            "hybrid": ["textrank-bart", "lexrank-t5"]
        }
