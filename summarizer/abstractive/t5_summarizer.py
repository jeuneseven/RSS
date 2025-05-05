# summarizer/abstractive/t5_summarizer.py
from summarizer.abstractive.base_abstractive import BaseAbstractiveSummarizer
from typing import Dict, Any, Optional
from transformers import pipeline


class T5Summarizer(BaseAbstractiveSummarizer):
    """
    T5 model implementation for abstractive summarization.
    Uses the Hugging Face transformers library.
    """

    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize T5 summarizer

        Args:
            model_name: Name of the pre-trained T5 model to use
        """
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=model_name)

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with T5 specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "T5",
            "description": "Abstractive summarization using the T5 model",
            "model_name": self.model_name
        })
        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the T5 model"""
        return {
            "model_name": self.model_name,
            "model_type": "T5",
            "language": "english"
        }

    def summarize(self, text: str, max_length: int = 100, min_length: int = 30, **kwargs) -> str:
        """
        Generate an abstractive summary using T5 model

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
            # T5 requires prefix for summarization task
            prefixed_text = "summarize: " + text

            # T5 has input token limits
            max_input_chars = 1024  # Conservative limit
            if len(prefixed_text) > max_input_chars:
                prefixed_text = prefixed_text[:max_input_chars]
                print(
                    f"Text truncated to {max_input_chars} characters for T5 model")

            # Set up summarization parameters
            params = {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": kwargs.get("do_sample", False),
                "num_beams": kwargs.get("num_beams", 4),
                "early_stopping": kwargs.get("early_stopping", True)
            }

            # Generate summary
            summary = self.summarizer(
                prefixed_text, **params)[0]['summary_text']

            return summary

        except Exception as e:
            print(f"Error in T5 summarization: {e}")
            # Fallback to simple extraction of first few sentences
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            sentences = nltk.sent_tokenize(text)
            return " ".join(sentences[:3])
