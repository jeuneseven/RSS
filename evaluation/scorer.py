"""
evaluation/scorer.py

Unified scoring interface for ROUGE and BERTScore.
- Ensures consistent evaluation for all summarization pipelines.
- Supports batch (multi-article) evaluation and averaging.
- Results are ready for JSON output.
"""

from rouge import Rouge
from bert_score import score as bert_score
import numpy as np


class Evaluator:
    def __init__(self, rouge_types=None, bert_lang='en'):
        """
        Unified scorer for summarization evaluation.
        :param rouge_types: List of ROUGE metrics to compute (default: ['rouge-1', 'rouge-2', 'rouge-l'])
        :param bert_lang: Language for BERTScore (default: 'en')
        """
        self.rouge_types = rouge_types if rouge_types else [
            'rouge-1', 'rouge-2', 'rouge-l']
        self.rouge = Rouge(metrics=self.rouge_types)
        self.bert_lang = bert_lang

    def score(self, summary, reference):
        """
        Compute ROUGE and BERTScore for a single summary/reference pair.
        :return: dict with ROUGE and BERTScore results
        """
        # Compute ROUGE
        rouge_scores = self.rouge.get_scores(summary, reference)[0]
        # Compute BERTScore
        P, R, F1 = bert_score([summary], [reference], lang=self.bert_lang)
        bertscore_result = {
            'p': P.item(),
            'r': R.item(),
            'f1': F1.item()
        }
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
                vals = [r['rouge'][key][stat] for r in result_list]
                avg[f'rouge_{key}_{stat}'] = float(
                    np.mean(vals)) if vals else 0.0
        # BERTScore: average p/r/f1
        for stat in ['p', 'r', 'f1']:
            vals = [r['bertscore'][stat] for r in result_list]
            avg[f'bertscore_{stat}'] = float(np.mean(vals)) if vals else 0.0
        return avg
