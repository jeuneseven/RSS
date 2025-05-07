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
        Optimized hybrid summarization method that combines extractive and abstractive approaches
        with enhanced sentence preservation and importance signaling

        Args:
            text: Input text to summarize
            **kwargs: Additional parameters including extractive_sentences, max_length, etc.

        Returns:
            Hybrid summary with improved ROUGE scores
        """
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."

        try:
            # 1. Extract important sentences and retain their importance scores
            sentences_count = kwargs.get("extractive_sentences", 10)
            ranked_sentences = self.extractive_summarizer.get_ranked_sentences(
                text)

            # 2. Select top N sentences but maintain original document order
            if len(ranked_sentences) > sentences_count:
                # Get indices of top N sentences
                top_indices = [s["index"]
                               for s in ranked_sentences[:sentences_count]]
                # Sort indices to maintain original document order
                top_indices.sort()
                # Rebuild extractive summary preserving original sentence order
                from nltk.tokenize import sent_tokenize
                all_sentences = sent_tokenize(text)
                extractive_summary = " ".join(
                    [all_sentences[i] for i in top_indices])
            else:
                extractive_summary = " ".join(
                    [s["text"] for s in ranked_sentences])

            # 3. Generate summary while ensuring inclusion of most important sentences
            most_important_sentence = ranked_sentences[0]["text"] if ranked_sentences else ""

            # Add prompt to the generative model emphasizing important information
            enhanced_input = f"Important information: {most_important_sentence}\n\nText to summarize: {extractive_summary}"

            max_length = kwargs.get("max_length", 100)
            min_length = kwargs.get("min_length", 30)

            abstractive_params = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": kwargs.get("num_beams", 4),
                "early_stopping": kwargs.get("early_stopping", True)
            }

            final_summary = self.abstractive_summarizer.summarize(
                enhanced_input, **abstractive_params)

            return final_summary

        except Exception as e:
            print(f"Error in hybrid summarization: {e}")
            return self.extractive_summarizer.summarize(text, sentences_count=3)
