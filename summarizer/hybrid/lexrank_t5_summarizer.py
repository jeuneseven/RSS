# summarizer/hybrid/lexrank_t5_summarizer.py
from summarizer.hybrid.base_hybrid import BaseHybridSummarizer
from summarizer.extractive.lexrank_summarizer import LexRankSummarizer
from summarizer.abstractive.t5_summarizer import T5Summarizer
from typing import Dict, Any, Optional


class LexRankT5Summarizer(BaseHybridSummarizer):
    """
    Hybrid summarizer that uses LexRank to extract important sentences,
    then applies T5 to generate an abstractive summary from those extracts.
    """

    def __init__(self, lexrank_sentences: int = 10,
                 extractive_summarizer=None,
                 abstractive_summarizer=None):
        """
        Initialize LexRank → T5 hybrid summarizer

        Args:
            lexrank_sentences: Number of sentences to extract with LexRank
            extractive_summarizer: LexRank summarizer instance (created if None)
            abstractive_summarizer: T5 summarizer instance (created if None)
        """
        # Create default summarizers if not provided
        if extractive_summarizer is None:
            extractive_summarizer = LexRankSummarizer()

        if abstractive_summarizer is None:
            abstractive_summarizer = T5Summarizer()

        super().__init__(extractive_summarizer, abstractive_summarizer)
        self.lexrank_sentences = lexrank_sentences

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with hybrid specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "LexRank→T5",
            "description": "Hybrid summarization that extracts sentences with LexRank then applies T5",
            "extractive_sentences": self.lexrank_sentences
        })
        return metadata

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30, **kwargs) -> str:
        """
        Generate a hybrid summary by first using LexRank, then T5

        Args:
            text: Input text to summarize
            max_length: Maximum length of the final summary
            min_length: Minimum length of the final summary
            **kwargs: Additional parameters

        Returns:
            Hybrid summary
        """
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."

        try:
            # Step 1: Extract sentences with LexRank
            sentences_count = kwargs.get(
                "lexrank_sentences", self.lexrank_sentences)
            extractive_summary = self.extractive_summarizer.summarize(
                text, sentences_count=sentences_count)

            # Step 2: Generate abstractive summary from the extracted sentences
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
            print(f"Error in LexRank→T5 summarization: {e}")
            # Fallback to simple extractive summary
            return self.extractive_summarizer.summarize(text, sentences_count=3)
