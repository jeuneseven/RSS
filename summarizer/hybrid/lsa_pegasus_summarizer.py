# summarizer/hybrid/lsa_pegasus_summarizer.py
from summarizer.hybrid.base_hybrid import BaseHybridSummarizer
from summarizer.extractive.lsa_summarizer import LSASummarizer
from summarizer.abstractive.pegasus_summarizer import PegasusSummarizer
from typing import Dict, Any, Optional


class LSAPegasusSummarizer(BaseHybridSummarizer):
    """
    Hybrid summarizer that uses LSA to extract important sentences,
    then applies Pegasus to generate an abstractive summary from those extracts.
    """

    def __init__(self, lsa_sentences: int = 10,
                 extractive_summarizer=None,
                 abstractive_summarizer=None):
        """
        Initialize LSA → Pegasus hybrid summarizer

        Args:
            lsa_sentences: Number of sentences to extract with LSA
            extractive_summarizer: LSA summarizer instance (created if None)
            abstractive_summarizer: Pegasus summarizer instance (created if None)
        """
        # Create default summarizers if not provided
        if extractive_summarizer is None:
            extractive_summarizer = LSASummarizer()

        if abstractive_summarizer is None:
            abstractive_summarizer = PegasusSummarizer()

        super().__init__(extractive_summarizer, abstractive_summarizer)
        self.lsa_sentences = lsa_sentences

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with hybrid specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "LSA→Pegasus",
            "description": "Hybrid summarization that extracts sentences with LSA then applies Pegasus",
            "extractive_sentences": self.lsa_sentences
        })
        return metadata

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30, **kwargs) -> str:
        """
        Generate a hybrid summary by first using LSA, then Pegasus

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
            # Step 1: Extract sentences with LSA
            sentences_count = kwargs.get("lsa_sentences", self.lsa_sentences)
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
            print(f"Error in LSA→Pegasus summarization: {e}")
            # Fallback to simple extractive summary
            return self.extractive_summarizer.summarize(text, sentences_count=3)
