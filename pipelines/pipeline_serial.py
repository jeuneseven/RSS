"""
pipeline_serial.py (Enhanced with intelligent extractive combination)
- Upgraded combine mode to use semantic deduplication like parallel pipeline
- Maintains compatibility with existing interface and evaluation structure
"""

import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class SerialPipeline:
    def __init__(self, extractive_method='textrank', abstractive_method='bart',
                 max_length=150, min_length=50, num_beams=4, device=None):
        self.extractive_method = extractive_method
        self.abstractive_method = abstractive_method
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

        # Initialize stopwords for intelligent combination
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

    def _sentence_similarity(self, sent1, sent2):
        """
        Calculate semantic similarity between two sentences using cosine similarity.
        Enhanced from parallel pipeline for intelligent deduplication.

        Args:
            sent1: First sentence
            sent2: Second sentence

        Returns:
            Float between 0 and 1 representing similarity
        """
        if not sent1.strip() or not sent2.strip():
            return 0.0

        if sent1.strip() == sent2.strip():
            return 1.0

        try:
            # Tokenize and filter words, removing stopwords and non-alphanumeric tokens
            words1 = [word.lower() for word in word_tokenize(sent1)
                      if word.lower() not in self.stop_words and word.isalnum()]
            words2 = [word.lower() for word in word_tokenize(sent2)
                      if word.lower() not in self.stop_words and word.isalnum()]

            if not words1 or not words2:
                return 0.0

            # Create vocabulary from both sentences
            all_words = list(set(words1 + words2))

            # Create binary word vectors
            vector1 = [1 if word in words1 else 0 for word in all_words]
            vector2 = [1 if word in words2 else 0 for word in all_words]

            if not any(vector1) or not any(vector2):
                return 0.0

            # Calculate cosine similarity
            similarity = 1 - cosine_distance(vector1, vector2)
            return max(0.0, min(1.0, similarity))

        except Exception:
            # Fallback to simple string comparison in case of error
            return 1.0 if sent1.strip().lower() == sent2.strip().lower() else 0.0

    def combine_extractive_summaries(self, tr_sum, lr_sum, lsa_sum):
        """
        Intelligently combine three extractive summaries with semantic deduplication.
        Enhanced version replacing simple string concatenation.

        Args:
            tr_sum: TextRank summary
            lr_sum: LexRank summary
            lsa_sum: LSA summary

        Returns:
            Combined and deduplicated extractive summary
        """
        # Combine all sentences from the three summaries
        all_sentences = []
        all_sentences.extend(sent_tokenize(tr_sum))
        all_sentences.extend(sent_tokenize(lr_sum))
        all_sentences.extend(sent_tokenize(lsa_sum))

        # Remove empty sentences
        all_sentences = [sent.strip()
                         for sent in all_sentences if sent.strip()]

        if not all_sentences:
            return ""

        # Semantic deduplication using similarity threshold
        unique_sentences = []
        similarity_threshold = 0.8  # Sentences with >80% similarity are considered duplicates

        for sent in all_sentences:
            is_unique = True
            for existing_sent in unique_sentences:
                if self._sentence_similarity(sent, existing_sent) > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_sentences.append(sent)

        # Limit to reasonable length (7 sentences max for abstractive model input)
        max_sentences = 7
        final_sentences = unique_sentences[:max_sentences]

        return " ".join(final_sentences)

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, combine=False):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]
        ex_summaries, ab_summaries, hy_summaries = [], [], []
        tr_summaries, lr_summaries, lsa_summaries = [], [], []

        for i, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            content = article['content']

            # ADD: Store cleaned original content
            item['original_content'] = content

            if combine:
                # Enhanced combine mode: run all extractive methods and intelligently combine
                sum_tr = self.extractive.textrank(content)
                sum_lr = self.extractive.lexrank(content)
                sum_lsa = self.extractive.lsa(content)
                tr_summaries.append(sum_tr)
                lr_summaries.append(sum_lr)
                lsa_summaries.append(sum_lsa)

                # Use intelligent combination instead of simple concatenation
                ext_summary = self.combine_extractive_summaries(
                    sum_tr, sum_lr, sum_lsa)
            else:
                # Regular mode: use only selected extractive method
                sum_tr = sum_lr = sum_lsa = None
                ext_summary = getattr(
                    self.extractive, self.extractive_method)(content)

            # Generate abstractive summary
            abs_summary = getattr(
                self.abstractive, self.abstractive_method)(content)

            # Generate hybrid summary using combined/selected extractive as input
            if combine:
                # Use intelligently combined extractive summary as hybrid input
                hybrid_prompt = ext_summary
            else:
                # Use selected extractive summary as hybrid input
                hybrid_prompt = ext_summary

            hybrid_summary = getattr(
                self.abstractive, self.abstractive_method)(hybrid_prompt)

            # Store results and evaluate
            item['extractive_summary'] = ext_summary
            item['abstractive_summary'] = abs_summary
            item['hybrid_summary'] = hybrid_summary
            item['extractive_scores'] = self.evaluator.score(
                ext_summary, content)
            item['abstractive_scores'] = self.evaluator.score(
                abs_summary, content)
            item['hybrid_scores'] = self.evaluator.score(
                hybrid_summary, content)

            if combine:
                # Store individual extractive summaries and scores for plotting
                item['textrank_summary'] = sum_tr
                item['lexrank_summary'] = sum_lr
                item['lsa_summary'] = sum_lsa
                item['textrank_scores'] = self.evaluator.score(sum_tr, content)
                item['lexrank_scores'] = self.evaluator.score(sum_lr, content)
                item['lsa_scores'] = self.evaluator.score(sum_lsa, content)

            ex_summaries.append(ext_summary)
            ab_summaries.append(abs_summary)
            hy_summaries.append(hybrid_summary)
            results_json.append(item)

        # Batch evaluation for average scores
        avg_ex = self.evaluator.batch_score(ex_summaries, references)[1]
        avg_ab = self.evaluator.batch_score(ab_summaries, references)[1]
        avg_hy = self.evaluator.batch_score(hy_summaries, references)[1]
        avg_tr = self.evaluator.batch_score(tr_summaries, references)[
            1] if combine else {}
        avg_lr = self.evaluator.batch_score(lr_summaries, references)[
            1] if combine else {}
        avg_lsa = self.evaluator.batch_score(lsa_summaries, references)[
            1] if combine else {}

        # Prepare output structure
        output = {
            'articles': results_json,
            'average_scores': {
                'extractive': avg_ex,  # Combined extractive scores in combine mode
                'abstractive': avg_ab,
                'hybrid': avg_hy,      # Hybrid using intelligent combination
                # Individual method scores (combine mode only)
                'textrank': avg_tr,
                'lexrank': avg_lr,
                'lsa': avg_lsa
            }
        }

        # Save results with unique naming based on combine mode
        if combine:
            json_filename = f'serial_combined_{self.abstractive_method}.json'
        else:
            json_filename = f'serial_{self.extractive_method}_{self.abstractive_method}.json'

        json_path = os.path.join(outdir, json_filename)
        save_json(output, json_path)
        print(f'Serial pipeline JSON result saved to {json_path}')
        return output
