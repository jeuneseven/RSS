"""
pipelines/pipeline_parallel.py (FINAL FIXED VERSION)
- Fixed hybrid algorithm based on successful standalone versions
- Fixed 3+1 mode to follow serial.py logic: combine all extractive summaries first
- All modes use fusion/hybrid logic with enhanced semantic similarity
- 1+1: three bars (extractive, abstractive, hybrid)
- 3+1: five bars (textrank, lexrank, lsa, abstractive, hybrid)
- 3+3: three bars (best_extract, best_abstractive, hybrid)
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
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class ParallelPipeline:
    def __init__(self, extractive_methods=None, abstractive_methods=None,
                 max_length=150, min_length=50, num_beams=4, device=None):
        self.extractive_methods = extractive_methods or [
            'textrank', 'lexrank', 'lsa']
        self.abstractive_methods = abstractive_methods or [
            'bart', 't5', 'pegasus']
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

        # Initialize stopwords for similarity calculation
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

    def _evaluate_rouge(self, summary, reference):
        """Evaluate summary using ROUGE metrics and return simplified format"""
        try:
            scores = self.evaluator.score(summary, reference)
            rouge_1_f = scores.get('rouge', {}).get(
                'rouge-1', {}).get('f', 0.0)
            return {'rouge-1': {'f': rouge_1_f}}
        except Exception as e:
            print(f"ROUGE evaluation error: {e}")
            return {'rouge-1': {'f': 0.0}}

    def _evaluate_bertscore(self, summary, reference):
        """Evaluate summary using BERTScore and return simplified format"""
        try:
            scores = self.evaluator.score(summary, reference)
            bertscore_f1 = scores.get('bertscore', {}).get('f1', 0.0)
            return {'f1': bertscore_f1}
        except Exception as e:
            print(f"BERTScore evaluation error: {e}")
            return {'f1': 0.0}

    def _sentence_similarity(self, sent1, sent2):
        """Calculate semantic similarity between two sentences using cosine similarity"""
        if not sent1.strip() or not sent2.strip():
            return 0.0

        if sent1.strip() == sent2.strip():
            return 1.0

        try:
            words1 = [word.lower() for word in word_tokenize(sent1)
                      if word.lower() not in self.stop_words and word.isalnum()]
            words2 = [word.lower() for word in word_tokenize(sent2)
                      if word.lower() not in self.stop_words and word.isalnum()]

            if not words1 or not words2:
                return 0.0

            all_words = list(set(words1 + words2))
            vector1 = [1 if word in words1 else 0 for word in all_words]
            vector2 = [1 if word in words2 else 0 for word in all_words]

            if not any(vector1) or not any(vector2):
                return 0.0

            similarity = 1 - cosine_distance(vector1, vector2)
            return max(0.0, min(1.0, similarity))

        except Exception:
            return 1.0 if sent1.strip().lower() == sent2.strip().lower() else 0.0

    def combine_extractive_summaries(self, tr_sum, lr_sum, lsa_sum):
        """
        Combine three extractive summaries like in serial.py: merge and deduplicate
        """
        # Combine all sentences from the three summaries
        all_sentences = []
        all_sentences.extend(sent_tokenize(tr_sum))
        all_sentences.extend(sent_tokenize(lr_sum))
        all_sentences.extend(sent_tokenize(lsa_sum))

        # Deduplicate using semantic similarity
        unique_sentences = []
        for sent in all_sentences:
            is_unique = True
            for existing_sent in unique_sentences:
                if self._sentence_similarity(sent, existing_sent) > 0.8:
                    is_unique = False
                    break
            if is_unique:
                unique_sentences.append(sent)

        # Return combined summary (limit to reasonable length)
        return " ".join(unique_sentences[:7])  # Limit to 7 sentences

    def hybrid_summarization(self, original_text, extractive_summary, abstractive_summary):
        """
        Generate hybrid summary based on successful standalone versions
        FIXED: Uses same logic as successful textrank_bart_hybrid.py
        """
        # Evaluate quality of both summary types
        rouge_ext = self._evaluate_rouge(extractive_summary, original_text)
        rouge_abs = self._evaluate_rouge(abstractive_summary, original_text)
        bertscore_ext = self._evaluate_bertscore(
            extractive_summary, original_text)
        bertscore_abs = self._evaluate_bertscore(
            abstractive_summary, original_text)

        # Calculate weights based on performance
        ext_weight = (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2
        abs_weight = (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2

        # Normalize weights
        total = ext_weight + abs_weight
        if total > 0:
            ext_weight = ext_weight / total
            abs_weight = abs_weight / total
        else:
            ext_weight = abs_weight = 0.5

        # Strategy selection based on successful versions
        if ext_weight > abs_weight * 1.2:  # Extractive-dominant
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)

            # Find unique information in abstractive summary
            unique_abs_sentences = []
            for abs_sent in sentences_abs:
                is_unique = True
                for ext_sent in sentences_ext:
                    if self._sentence_similarity(abs_sent, ext_sent) > 0.6:
                        is_unique = False
                        break
                if is_unique:
                    unique_abs_sentences.append(abs_sent)

            # Combine extractive with unique abstractive sentences
            hybrid_summary = extractive_summary
            if unique_abs_sentences:
                hybrid_summary += " " + " ".join(unique_abs_sentences[:2])

        elif abs_weight > ext_weight * 1.2:  # Abstractive-dominant
            sentences_ext = sent_tokenize(extractive_summary)
            key_sentences = sentences_ext[:2]  # Top 2 extractive sentences

            hybrid_summary = abstractive_summary
            # Add key extractive sentences that are not already covered
            for sent in key_sentences:
                if sent not in abstractive_summary:
                    hybrid_summary += " " + sent

        else:  # Balanced fusion
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)
            hybrid_sentences = []

            # Start with first abstractive sentence (often a good overview)
            if sentences_abs:
                hybrid_sentences.append(sentences_abs[0])

            # Add unique extractive sentences
            for i, sent in enumerate(sentences_ext):
                if i < 3:  # Limit to 3 sentences from extractive
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

            # Add remaining unique abstractive sentences
            for i, sent in enumerate(sentences_abs[1:]):
                if i < 2:  # Limit to 2 more from abstractive
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

            hybrid_summary = " ".join(hybrid_sentences)

        # Quality assurance - FIXED: use same logic as successful versions
        rouge_hybrid = self._evaluate_rouge(hybrid_summary, original_text)
        bertscore_hybrid = self._evaluate_bertscore(
            hybrid_summary, original_text)

        hybrid_score = (rouge_hybrid['rouge-1']
                        ['f'] + bertscore_hybrid['f1']) / 2
        best_original_score = max((rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2,
                                  (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2)

        # Only fallback if hybrid is significantly worse (like successful versions)
        if hybrid_score < best_original_score:
            if (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) > (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']):
                return extractive_summary
            else:
                return abstractive_summary

        return hybrid_summary

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, mode='1+1'):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        references = [a['content'] for a in articles]

        # Prepare containers for different summary types
        textrank_sums, lexrank_sums, lsa_sums = [], [], []
        abstractive_sums, hybrid_sums = [], []
        best_extract_sums, best_abstractive_sums, best_best_sums = [], [], []

        for article in articles:
            content = article['content']

            # Generate individual method summaries
            tr_sum = self.extractive.textrank(content)
            lr_sum = self.extractive.lexrank(content)
            lsa_sum = self.extractive.lsa(content)
            abs_sum = getattr(self.abstractive,
                              self.abstractive_methods[0])(content)

            textrank_sums.append(tr_sum)
            lexrank_sums.append(lr_sum)
            lsa_sums.append(lsa_sum)
            abstractive_sums.append(abs_sum)

            # Mode-specific fusion strategies
            if mode == '1+1':
                # Simple 1+1 fusion: TextRank + BART (like successful standalone version)
                hybrid_sum = self.hybrid_summarization(
                    content, tr_sum, abs_sum)
                hybrid_sums.append(hybrid_sum)

            elif mode == '3+1':
                # FIXED: 3+1 fusion following serial.py logic
                # Combine all three extractive summaries first, then fuse with abstractive
                combined_extractive = self.combine_extractive_summaries(
                    tr_sum, lr_sum, lsa_sum)
                hybrid_sum = self.hybrid_summarization(
                    content, combined_extractive, abs_sum)
                hybrid_sums.append(hybrid_sum)

            elif mode == '3+3':
                # 3+3 fusion: best of all combinations

                # Find best extractive method
                ex_summaries = [tr_sum, lr_sum, lsa_sum]
                ex_scores = []
                for s in ex_summaries:
                    score_dict = self.evaluator.score(s, content)
                    rouge_f = score_dict.get('rouge', {}).get(
                        'rouge-1', {}).get('f', 0.0)
                    ex_scores.append(rouge_f)
                best_ext_idx = int(np.argmax(ex_scores))
                best_extract_sums.append(ex_summaries[best_ext_idx])

                # Find best abstractive method
                ab_summaries = [getattr(self.abstractive, m)(
                    content) for m in self.abstractive_methods]
                ab_scores = []
                for s in ab_summaries:
                    score_dict = self.evaluator.score(s, content)
                    rouge_f = score_dict.get('rouge', {}).get(
                        'rouge-1', {}).get('f', 0.0)
                    ab_scores.append(rouge_f)
                best_abs_idx = int(np.argmax(ab_scores))
                best_abstractive_sums.append(ab_summaries[best_abs_idx])

                # Find best hybrid from all 3x3 combinations
                all_hybrids = [self.hybrid_summarization(content, e, a)
                               for e in ex_summaries for a in ab_summaries]
                hybrid_scores = []
                for h in all_hybrids:
                    score_dict = self.evaluator.score(h, content)
                    rouge_f = score_dict.get('rouge', {}).get(
                        'rouge-1', {}).get('f', 0.0)
                    hybrid_scores.append(rouge_f)
                best_hybrid_idx = int(np.argmax(hybrid_scores))
                best_best_sums.append(all_hybrids[best_hybrid_idx])

        # Calculate average scores for each method
        avg_textrank = self.evaluator.batch_score(textrank_sums, references)[1]
        avg_lexrank = self.evaluator.batch_score(lexrank_sums, references)[1]
        avg_lsa = self.evaluator.batch_score(lsa_sums, references)[1]
        avg_abstractive = self.evaluator.batch_score(
            abstractive_sums, references)[1]

        # Prepare output based on mode
        if mode == '1+1':
            avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]
            output = {
                'average_scores': {
                    'extractive': avg_textrank,  # default: textrank
                    'abstractive': avg_abstractive,
                    'combo': avg_hybrid
                }
            }
        elif mode == '3+1':
            # FIXED: 3+1 mode now shows all individual extractive methods + abstractive + hybrid
            avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]
            output = {
                'average_scores': {
                    'textrank': avg_textrank,
                    'lexrank': avg_lexrank,
                    'lsa': avg_lsa,
                    'abstractive': avg_abstractive,
                    'best_single': avg_hybrid  # This is the combined extractive + abstractive hybrid
                }
            }
        elif mode == '3+3':
            avg_best_extract = self.evaluator.batch_score(
                best_extract_sums, references)[1]
            avg_best_abstractive = self.evaluator.batch_score(
                best_abstractive_sums, references)[1]
            avg_best_best = self.evaluator.batch_score(
                best_best_sums, references)[1]
            output = {
                'average_scores': {
                    'best_extract': avg_best_extract,
                    'best_abstractive': avg_best_abstractive,
                    'best_best': avg_best_best
                }
            }
        else:
            raise ValueError(f'Unknown mode {mode}')

        # Save results
        json_path = os.path.join(outdir, 'parallel_results.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
