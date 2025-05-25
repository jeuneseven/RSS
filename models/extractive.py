"""
models/extractive.py

Unified interface for extractive summarization algorithms: TextRank, LexRank, LSA
All parameters are managed centrally to ensure consistency across the project.
Algorithms are called as black boxes using reliable external libraries (Sumy, sklearn, etc.).
"""

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
import numpy as np

nltk.download('punkt', quiet=True)


class ExtractiveSummarizer:
    def __init__(self, language='english', num_sentences=5):
        """
        ExtractiveSummarizer manages all extractive summarization algorithms with unified parameters.
        :param language: Language for sentence tokenization (default: 'english')
        :param num_sentences: Default number of sentences in summary (default: 5)
        """
        self.language = language
        self.num_sentences = num_sentences

    def textrank(self, text, num_sentences=None):
        """
        TextRank extractive summarization using sumy library.
        :param text: Article text
        :param num_sentences: Number of sentences for the summary (if None, use default)
        :return: Summarized text (string)
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join(str(sentence) for sentence in summary)

    def lexrank(self, text, num_sentences=None):
        """
        LexRank extractive summarization using sumy library.
        :param text: Article text
        :param num_sentences: Number of sentences for the summary (if None, use default)
        :return: Summarized text (string)
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join(str(sentence) for sentence in summary)

    def lsa(self, text, num_sentences=None, n_components=None):
        """
        LSA extractive summarization using sklearn.
        :param text: Article text
        :param num_sentences: Number of sentences for the summary (if None, use default)
        :param n_components: Number of SVD topics/components (if None, use 5 or less)
        :return: Summarized text (string)
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        tfidf = TfidfVectorizer(stop_words=self.language)
        X = tfidf.fit_transform(sentences)
        if n_components is None:
            n_components = min(5, X.shape[0]-1, X.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)
        scores = np.zeros(len(sentences))
        for i in range(len(sentences)):
            vec = X[i].toarray()[0]
            for component, sigma in zip(svd.components_, svd.singular_values_):
                scores[i] += abs(np.dot(vec, component)) * sigma
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :num_sentences]
        return ' '.join([sentences[i] for i in sorted(ranked)])
