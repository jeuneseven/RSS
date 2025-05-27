"""
evaluation/scorer.py (FIXED VERSION)

Fixed scoring interface for ROUGE and BERTScore with better differentiation.
Drop-in replacement for the original scorer.py that addresses ROUGE-1/ROUGE-L similarity issues.
"""

from rouge import Rouge
from bert_score import score as bert_score
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class Evaluator:
    def __init__(self, rouge_types=None, bert_lang='en'):
        """
        Fixed scorer for summarization evaluation.
        :param rouge_types: List of ROUGE metrics to compute (default: ['rouge-1', 'rouge-2', 'rouge-l'])
        :param bert_lang: Language for BERTScore (default: 'en')
        """
        self.rouge_types = rouge_types if rouge_types else [
            'rouge-1', 'rouge-2', 'rouge-l']
        self.rouge = Rouge(metrics=self.rouge_types)
        self.bert_lang = bert_lang

    def _preprocess_text(self, text):
        """
        Enhanced text preprocessing for consistent evaluation.
        """
        if not text or not isinstance(text, str):
            return ""

        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())

        # Ensure proper sentence ending
        if text and not text[-1] in '.!?':
            text += '.'

        # Remove problematic characters that might interfere with ROUGE
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _manual_rouge_l(self, summary, reference):
        """
        Manual ROUGE-L calculation with improved precision.
        """
        try:
            # Tokenize
            summary_tokens = word_tokenize(summary.lower())
            reference_tokens = word_tokenize(reference.lower())

            # Remove very short tokens and punctuation
            summary_tokens = [
                t for t in summary_tokens if len(t) > 1 and t.isalnum()]
            reference_tokens = [
                t for t in reference_tokens if len(t) > 1 and t.isalnum()]

            if not summary_tokens or not reference_tokens:
                return {'f': 0.0, 'p': 0.0, 'r': 0.0}

            # Calculate LCS using dynamic programming
            m, n = len(summary_tokens), len(reference_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if summary_tokens[i-1] == reference_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            lcs_length = dp[m][n]

            # Calculate precision, recall, F1
            precision = lcs_length / len(summary_tokens)
            recall = lcs_length / len(reference_tokens)
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0.0

            return {'f': f1, 'p': precision, 'r': recall}

        except Exception:
            return {'f': 0.0, 'p': 0.0, 'r': 0.0}

    def score(self, summary, reference):
        """
        Compute ROUGE and BERTScore for a single summary/reference pair.
        Enhanced version with better ROUGE-L calculation.
        """
        # Preprocess texts
        proc_summary = self._preprocess_text(summary)
        proc_reference = self._preprocess_text(reference)

        if not proc_summary.strip() or not proc_reference.strip():
            # Return zero scores for empty inputs
            rouge_scores = {rt: {'f': 0.0, 'p': 0.0, 'r': 0.0}
                            for rt in self.rouge_types}
            bertscore_result = {'p': 0.0, 'r': 0.0, 'f1': 0.0}
        else:
            try:
                # Compute ROUGE using library
                rouge_scores = self.rouge.get_scores(
                    proc_summary, proc_reference)[0]

                # Replace ROUGE-L with manual calculation for better precision
                if 'rouge-l' in rouge_scores:
                    manual_rl = self._manual_rouge_l(
                        proc_summary, proc_reference)
                    rouge_scores['rouge-l'] = manual_rl

            except Exception:
                # Fallback to manual calculations for all metrics
                rouge_scores = {}

                # Manual ROUGE-1
                try:
                    summary_words = set(word_tokenize(proc_summary.lower()))
                    reference_words = set(
                        word_tokenize(proc_reference.lower()))
                    summary_words = {
                        w for w in summary_words if len(w) > 1 and w.isalnum()}
                    reference_words = {
                        w for w in reference_words if len(w) > 1 and w.isalnum()}

                    if reference_words:
                        overlap = len(summary_words & reference_words)
                        precision = overlap / \
                            len(summary_words) if summary_words else 0.0
                        recall = overlap / len(reference_words)
                        f1 = 2 * precision * recall / \
                            (precision + recall) if (precision + recall) > 0 else 0.0
                        rouge_scores['rouge-1'] = {'f': f1,
                                                   'p': precision, 'r': recall}
                    else:
                        rouge_scores['rouge-1'] = {'f': 0.0,
                                                   'p': 0.0, 'r': 0.0}
                except Exception:
                    rouge_scores['rouge-1'] = {'f': 0.0, 'p': 0.0, 'r': 0.0}

                # Manual ROUGE-2
                try:
                    from collections import defaultdict
                    summary_tokens = word_tokenize(proc_summary.lower())
                    reference_tokens = word_tokenize(proc_reference.lower())

                    if len(summary_tokens) > 1 and len(reference_tokens) > 1:
                        summary_bigrams = set(
                            zip(summary_tokens[:-1], summary_tokens[1:]))
                        reference_bigrams = set(
                            zip(reference_tokens[:-1], reference_tokens[1:]))

                        overlap = len(summary_bigrams & reference_bigrams)
                        precision = overlap / \
                            len(summary_bigrams) if summary_bigrams else 0.0
                        recall = overlap / \
                            len(reference_bigrams) if reference_bigrams else 0.0
                        f1 = 2 * precision * recall / \
                            (precision + recall) if (precision + recall) > 0 else 0.0
                        rouge_scores['rouge-2'] = {'f': f1,
                                                   'p': precision, 'r': recall}
                    else:
                        rouge_scores['rouge-2'] = {'f': 0.0,
                                                   'p': 0.0, 'r': 0.0}
                except Exception:
                    rouge_scores['rouge-2'] = {'f': 0.0, 'p': 0.0, 'r': 0.0}

                # Manual ROUGE-L
                rouge_scores['rouge-l'] = self._manual_rouge_l(
                    proc_summary, proc_reference)

            # Compute BERTScore
            try:
                P, R, F1 = bert_score(
                    [proc_summary], [proc_reference], lang=self.bert_lang)
                bertscore_result = {
                    'p': float(P.item()),
                    'r': float(R.item()),
                    'f1': float(F1.item())
                }
            except Exception:
                bertscore_result = {'p': 0.0, 'r': 0.0, 'f1': 0.0}

        return {
            'rouge': rouge_scores,
            'bertscore': bertscore_result
        }

    def batch_score(self, summaries, references):
        """
        Compute ROUGE/BERTScore for lists of summaries and references.
        Returns list of dicts (one per article), and also provides average.
        """
        assert len(summaries) == len(references)
        results = [self.score(s, r) for s, r in zip(summaries, references)]
        avg_result = self.average_scores(results)
        return results, avg_result

    def average_scores(self, result_list):
        """
        Average a list of score dicts (for batch evaluation output).
        :param result_list: List of {rouge: ..., bertscore: ...} dicts
        :return: Dict with averaged scores for each metric
        """
        avg = {}
        # ROUGE: average F1, P, R for each type
        for key in self.rouge_types:
            for stat in ['f', 'p', 'r']:
                try:
                    vals = [r['rouge'][key][stat]
                            for r in result_list if 'rouge' in r and key in r['rouge']]
                    avg[f'rouge_{key}_{stat}'] = float(
                        np.mean(vals)) if vals else 0.0
                except (KeyError, TypeError):
                    avg[f'rouge_{key}_{stat}'] = 0.0

        # BERTScore: average p/r/f1
        for stat in ['p', 'r', 'f1']:
            try:
                vals = [r['bertscore'][stat]
                        for r in result_list if 'bertscore' in r]
                avg[f'bertscore_{stat}'] = float(
                    np.mean(vals)) if vals else 0.0
            except (KeyError, TypeError):
                avg[f'bertscore_{stat}'] = 0.0

        return avg
