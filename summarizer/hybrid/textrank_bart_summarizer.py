# summarizer/hybrid/textrank_bart_summarizer.py
from summarizer.hybrid.base_hybrid import BaseHybridSummarizer
from summarizer.extractive.textrank_summarizer import TextRankSummarizer
from summarizer.abstractive.bart_summarizer import BARTSummarizer
from typing import Dict, Any, Optional


class TextRankBARTSummarizer(BaseHybridSummarizer):
    """
    Hybrid summarizer that uses TextRank to extract important sentences,
    then applies BART to generate an abstractive summary from those extracts.
    """

    def __init__(self, textrank_sentences: int = 10,
                 extractive_summarizer=None,
                 abstractive_summarizer=None):
        """
        Initialize TextRank → BART hybrid summarizer

        Args:
            textrank_sentences: Number of sentences to extract with TextRank
            extractive_summarizer: TextRank summarizer instance (created if None)
            abstractive_summarizer: BART summarizer instance (created if None)
        """
        # Create default summarizers if not provided
        if extractive_summarizer is None:
            extractive_summarizer = TextRankSummarizer()

        if abstractive_summarizer is None:
            abstractive_summarizer = BARTSummarizer()

        super().__init__(extractive_summarizer, abstractive_summarizer)
        self.textrank_sentences = textrank_sentences

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with hybrid specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "TextRank→BART",
            "description": "Hybrid summarization that extracts sentences with TextRank then applies BART",
            "extractive_sentences": self.textrank_sentences
        })
        return metadata

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30, **kwargs) -> str:
        """
        Generate a hybrid summary by first using TextRank, then BART

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
            # Step 1: Extract sentences with TextRank
            sentences_count = kwargs.get(
                "textrank_sentences", self.textrank_sentences)
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
            print(f"Error in TextRank→BART summarization: {e}")
            # Fallback to simple extractive summary
            return self.extractive_summarizer.summarize(text, sentences_count=3)
