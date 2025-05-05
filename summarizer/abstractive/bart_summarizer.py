# summarizer/abstractive/bart_summarizer.py
from summarizer.abstractive.base_abstractive import BaseAbstractiveSummarizer
from typing import Dict, Any, Optional
from transformers import pipeline


class BARTSummarizer(BaseAbstractiveSummarizer):
    """
    BART model implementation for abstractive summarization.
    Uses the Hugging Face transformers library.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-6-6"):
        """
        Initialize BART summarizer

        Args:
            model_name: Name of the pre-trained BART model to use
        """
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=model_name)

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with BART specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "BART",
            "description": "Abstractive summarization using the BART model",
            "model_name": self.model_name
        })
        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the BART model"""
        return {
            "model_name": self.model_name,
            "model_type": "BART",
            "language": "english"
        }

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30, **kwargs) -> str:
        """
        Generate an abstractive summary using BART model

        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            **kwargs: Additional parameters (e.g., num_beams, do_sample)

        Returns:
            Abstractive summary
        """
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."

        try:
            # BART has input token limits
            max_input_chars = 1024  # Conservative limit
            if len(text) > max_input_chars:
                text = text[:max_input_chars]
                print(
                    f"Text truncated to {max_input_chars} characters for BART model")

            # Set up summarization parameters
            params = {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": kwargs.get("do_sample", False),
                "num_beams": kwargs.get("num_beams", 4),
                "early_stopping": kwargs.get("early_stopping", True)
            }

            # Generate summary
            summary = self.summarizer(text, **params)[0]['summary_text']

            return summary

        except Exception as e:
            print(f"Error in BART summarization: {e}")
            # Fallback to simple extraction of first few sentences
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            sentences = nltk.sent_tokenize(text)
            return " ".join(sentences[:3])
