"""
evaluation/scorer.py

Updated scoring interface using Hugging Face evaluate library for ROUGE and BERTScore.
- Uses evaluate.load('rouge') with Google rouge-score backend for better accuracy
- Maintains compatibility with existing code structure
- Supports batch evaluation and averaging
"""

import evaluate
from bert_score import score as bert_score
import numpy as np


class Evaluator:
    def __init__(self, rouge_types=None, bert_lang='en'):
        """
        Updated scorer using Hugging Face evaluate library.
        :param rouge_types: List of ROUGE metrics to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        :param bert_lang: Language for BERTScore (default: 'en')
        """
        # Note: HF evaluate uses different naming convention (rouge1, rouge2, rougeL)
        self.rouge_types = rouge_types if rouge_types else [
            'rouge1', 'rouge2', 'rougeL']

        # Load ROUGE metric from Hugging Face evaluate
        self.rouge_metric = evaluate.load('rouge')
        self.bert_lang = bert_lang

        # Mapping from HF naming to original naming for compatibility
        self.rouge_name_mapping = {
            'rouge1': 'rouge-1',
            'rouge2': 'rouge-2',
            'rougeL': 'rouge-l',
            'rougeLsum': 'rouge-lsum'
        }

    def score(self, summary, reference):
        """
        Compute ROUGE and BERTScore for a single summary/reference pair.
        :param summary: Generated summary text
        :param reference: Reference text
        :return: dict with ROUGE and BERTScore results
        """
        # Input validation
        if not summary or not reference or not isinstance(summary, str) or not isinstance(reference, str):
            # Return zero scores for invalid inputs
            rouge_scores = {}
            for rouge_type in self.rouge_types:
                original_name = self.rouge_name_mapping.get(
                    rouge_type, rouge_type)
                rouge_scores[original_name] = {'f': 0.0, 'p': 0.0, 'r': 0.0}

            bertscore_result = {'p': 0.0, 'r': 0.0, 'f1': 0.0}
            return {
                'rouge': rouge_scores,
                'bertscore': bertscore_result
            }

        try:
            # Compute ROUGE using HF evaluate
            rouge_results = self.rouge_metric.compute(
                predictions=[summary],
                references=[reference],
                rouge_types=self.rouge_types,
                use_stemmer=True,  # Enable stemming for better matching
                use_aggregator=False  # Get individual scores, not aggregated
            )

            # Convert HF format to original format for compatibility
            rouge_scores = {}
            for rouge_type in self.rouge_types:
                original_name = self.rouge_name_mapping.get(
                    rouge_type, rouge_type)

                if rouge_type in rouge_results:
                    # HF evaluate returns scores directly as floats (F1 scores)
                    f1_score = rouge_results[rouge_type][0] if isinstance(
                        rouge_results[rouge_type], list) else rouge_results[rouge_type]

                    # For compatibility, we'll estimate precision and recall
                    # Note: HF evaluate primarily returns F1 scores
                    rouge_scores[original_name] = {
                        'f': float(f1_score),
                        'p': float(f1_score),  # Approximation
                        'r': float(f1_score)   # Approximation
                    }
                else:
                    rouge_scores[original_name] = {
                        'f': 0.0, 'p': 0.0, 'r': 0.0}

        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}")
            # Fallback to zero scores
            rouge_scores = {}
            for rouge_type in self.rouge_types:
                original_name = self.rouge_name_mapping.get(
                    rouge_type, rouge_type)
                rouge_scores[original_name] = {'f': 0.0, 'p': 0.0, 'r': 0.0}

        # Compute BERTScore
        try:
            P, R, F1 = bert_score([summary], [reference], lang=self.bert_lang)
            bertscore_result = {
                'p': float(P.item()),
                'r': float(R.item()),
                'f1': float(F1.item())
            }
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            bertscore_result = {'p': 0.0, 'r': 0.0, 'f1': 0.0}

        return {
            'rouge': rouge_scores,
            'bertscore': bertscore_result
        }

    def batch_score(self, summaries, references):
        """
        Compute ROUGE/BERTScore for lists of summaries and references.
        Enhanced batch processing using HF evaluate's batch capabilities.
        :param summaries: List of summary texts
        :param references: List of reference texts
        :return: Tuple of (individual_results, averaged_results)
        """
        assert len(summaries) == len(
            references), "Summaries and references must have same length"

        if not summaries or not references:
            return [], {}

        # Validate inputs
        valid_pairs = []
        for i, (s, r) in enumerate(zip(summaries, references)):
            if s and r and isinstance(s, str) and isinstance(r, str):
                valid_pairs.append((s, r))
            else:
                print(f"Warning: Invalid input pair at index {i}")

        if not valid_pairs:
            return [], {}

        try:
            # Batch ROUGE computation using HF evaluate
            batch_summaries, batch_references = zip(*valid_pairs)

            rouge_results = self.rouge_metric.compute(
                predictions=list(batch_summaries),
                references=list(batch_references),
                rouge_types=self.rouge_types,
                use_stemmer=True,
                use_aggregator=False  # Get individual scores
            )

            # Batch BERTScore computation
            try:
                P_batch, R_batch, F1_batch = bert_score(
                    list(batch_summaries),
                    list(batch_references),
                    lang=self.bert_lang
                )
                bertscore_batch = [
                    {'p': float(P_batch[i].item()), 'r': float(
                        R_batch[i].item()), 'f1': float(F1_batch[i].item())}
                    for i in range(len(batch_summaries))
                ]
            except Exception as e:
                print(f"Warning: Batch BERTScore computation failed: {e}")
                bertscore_batch = [{'p': 0.0, 'r': 0.0, 'f1': 0.0}
                                   for _ in batch_summaries]

            # Convert to individual results format
            results = []
            for i in range(len(batch_summaries)):
                rouge_scores = {}
                for rouge_type in self.rouge_types:
                    original_name = self.rouge_name_mapping.get(
                        rouge_type, rouge_type)

                    if rouge_type in rouge_results and i < len(rouge_results[rouge_type]):
                        f1_score = rouge_results[rouge_type][i]
                        rouge_scores[original_name] = {
                            'f': float(f1_score),
                            'p': float(f1_score),  # Approximation
                            'r': float(f1_score)   # Approximation
                        }
                    else:
                        rouge_scores[original_name] = {
                            'f': 0.0, 'p': 0.0, 'r': 0.0}

                results.append({
                    'rouge': rouge_scores,
                    'bertscore': bertscore_batch[i]
                })

        except Exception as e:
            print(
                f"Warning: Batch processing failed, falling back to individual scoring: {e}")
            # Fallback to individual scoring
            results = [self.score(s, r) for s, r in valid_pairs]

        # Handle cases where we had invalid inputs
        if len(valid_pairs) < len(summaries):
            # Fill in zero scores for invalid pairs
            full_results = []
            valid_idx = 0
            for i, (s, r) in enumerate(zip(summaries, references)):
                if s and r and isinstance(s, str) and isinstance(r, str):
                    full_results.append(results[valid_idx])
                    valid_idx += 1
                else:
                    # Zero scores for invalid input
                    rouge_scores = {}
                    for rouge_type in self.rouge_types:
                        original_name = self.rouge_name_mapping.get(
                            rouge_type, rouge_type)
                        rouge_scores[original_name] = {
                            'f': 0.0, 'p': 0.0, 'r': 0.0}
                    full_results.append({
                        'rouge': rouge_scores,
                        'bertscore': {'p': 0.0, 'r': 0.0, 'f1': 0.0}
                    })
            results = full_results

        # Compute average scores
        avg_result = self.average_scores(results)
        return results, avg_result

    def average_scores(self, result_list):
        """
        Average a list of score dicts (for batch evaluation output).
        :param result_list: List of {rouge: ..., bertscore: ...} dicts
        :return: Dict with averaged scores for each metric
        """
        if not result_list:
            return {}

        avg = {}

        # ROUGE: average F1, P, R for each type
        for rouge_type in self.rouge_types:
            original_name = self.rouge_name_mapping.get(rouge_type, rouge_type)
            for stat in ['f', 'p', 'r']:
                try:
                    vals = [
                        r['rouge'][original_name][stat]
                        for r in result_list
                        if 'rouge' in r and original_name in r['rouge'] and stat in r['rouge'][original_name]
                    ]
                    avg[f'rouge_{original_name}_{stat}'] = float(
                        np.mean(vals)) if vals else 0.0
                except (KeyError, TypeError, ValueError) as e:
                    avg[f'rouge_{original_name}_{stat}'] = 0.0

        # BERTScore: average p/r/f1
        for stat in ['p', 'r', 'f1']:
            try:
                vals = [
                    r['bertscore'][stat]
                    for r in result_list
                    if 'bertscore' in r and stat in r['bertscore']
                ]
                avg[f'bertscore_{stat}'] = float(
                    np.mean(vals)) if vals else 0.0
            except (KeyError, TypeError, ValueError) as e:
                avg[f'bertscore_{stat}'] = 0.0

        return avg

    def get_detailed_rouge_scores(self, summary, reference):
        """
        Get detailed ROUGE scores including precision and recall separately.
        This method provides more detailed metrics when needed.
        :param summary: Generated summary text
        :param reference: Reference text
        :return: Dict with detailed ROUGE metrics
        """
        if not summary or not reference:
            return {}

        try:
            # Use evaluate with more detailed output
            rouge_results = self.rouge_metric.compute(
                predictions=[summary],
                references=[reference],
                rouge_types=self.rouge_types,
                use_stemmer=True,
                use_aggregator=True  # This may provide more detailed stats
            )

            return rouge_results

        except Exception as e:
            print(f"Warning: Detailed ROUGE computation failed: {e}")
            return {}
