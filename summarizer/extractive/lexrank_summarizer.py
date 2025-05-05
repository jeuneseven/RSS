# summarizer/extractive/lexrank_summarizer.py
from summarizer.extractive.base_extractive import BaseExtractiveSummarizer
from typing import Dict, Any, List, Optional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRankSummarizer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class LexRankSummarizer(BaseExtractiveSummarizer):
    """
    LexRank algorithm implementation for extractive summarization.
    Uses the sumy library's implementation.
    """

    def __init__(self, language: str = "english"):
        """
        Initialize LexRank summarizer

        Args:
            language: Language for tokenization
        """
        self.language = language
        self.tokenizer = Tokenizer(language)
        self.summarizer = SumyLexRankSummarizer()

    def get_metadata(self) -> Dict[str, Any]:
        """Override metadata with LexRank specific information"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "LexRank",
            "description": "Graph-based extractive summarization using LexRank algorithm",
            "language": self.language
        })
        return metadata

    def summarize(self, text: str, sentences_count: int = 3, **kwargs) -> str:
        """
        Generate an extractive summary using LexRank algorithm

        Args:
            text: Input text to summarize
            sentences_count: Number of sentences to extract
            **kwargs: Additional parameters

        Returns:
            Extractive summary
        """
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."

        try:
            # Parse the text
            parser = PlaintextParser.from_string(text, self.tokenizer)

            # Apply LexRank algorithm
            summary = self.summarizer(
                parser.document, sentences_count=sentences_count)

            # Join the sentences into a string
            summary_text = " ".join(str(sentence) for sentence in summary)

            return summary_text

        except Exception as e:
            print(f"Error in LexRank summarization: {e}")
            # Fallback to first few sentences
            sentences = nltk.sent_tokenize(text)
            return " ".join(sentences[:sentences_count])

    def get_ranked_sentences(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get sentences ranked by LexRank importance score

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters

        Returns:
            List of dictionaries with sentence text and score
        """
        if not text or len(text.strip()) == 0:
            return []

        try:
            # Parse the text
            parser = PlaintextParser.from_string(text, self.tokenizer)

            # Get document
            document = parser.document

            # Get ratings
            ratings = self.summarizer.rate_sentences(document)

            # Create ranked list
            ranked_sentences = []
            for i, sentence in enumerate(document.sentences):
                score = ratings[i] if i in ratings else 0.0
                ranked_sentences.append({
                    "text": str(sentence),
                    "score": score,
                    "index": i
                })

            # Sort by score (descending)
            ranked_sentences.sort(key=lambda x: x["score"], reverse=True)

            return ranked_sentences

        except Exception as e:
            print(f"Error in LexRank sentence ranking: {e}")
            return []
