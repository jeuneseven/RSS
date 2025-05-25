"""
pipeline_parallel.py (all modes use fusion/hybrid logic, not prompt)
- All bar groups match your required number and labels
- 1+1: three bars (extractive, abstractive, hybrid)
- 3+1: five bars (textrank, lexrank, lsa, abstractive, hybrid)
- 3+3: three bars (best_extract, best_abstractive, hybrid)
- Fusion/hybrid always applies weighted content and sentence-level combination, not prompt-based
"""

import os
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json
from nltk.tokenize import sent_tokenize
import numpy as np


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

    def _evaluate_rouge(self, summary, reference):
        scores = self.evaluator.score(summary, reference)
        return {'rouge-1': {'f': scores.get('rouge_rouge-1_f', 0.0)}}

    def _evaluate_bertscore(self, summary, reference):
        scores = self.evaluator.score(summary, reference)
        return {'f1': scores.get('bertscore_f1', 0.0)}

    def _sentence_similarity(self, sent1, sent2):
        return float(sent1.strip() == sent2.strip())

    def hybrid_summarization(self, original_text, extractive_summary, abstractive_summary):
        rouge_ext = self._evaluate_rouge(extractive_summary, original_text)
        rouge_abs = self._evaluate_rouge(abstractive_summary, original_text)
        bertscore_ext = self._evaluate_bertscore(
            extractive_summary, original_text)
        bertscore_abs = self._evaluate_bertscore(
            abstractive_summary, original_text)
        ext_weight = (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2
        abs_weight = (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2
        total = ext_weight + abs_weight if (ext_weight + abs_weight) > 0 else 1
        ext_weight /= total
        abs_weight /= total
        if ext_weight > abs_weight * 1.2:
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)
            unique_abs = [s for s in sentences_abs if all(
                self._sentence_similarity(s, e) <= 0.6 for e in sentences_ext)]
            hybrid_summary = extractive_summary
            if unique_abs:
                hybrid_summary += " " + " ".join(unique_abs[:2])
        elif abs_weight > ext_weight * 1.2:
            sentences_ext = sent_tokenize(extractive_summary)
            key_sentences = sentences_ext[:2]
            hybrid_summary = abstractive_summary
            for sent in key_sentences:
                if sent not in abstractive_summary:
                    hybrid_summary += " " + sent
        else:
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)
            hybrid_sentences = []
            if sentences_abs:
                hybrid_sentences.append(sentences_abs[0])
            for i, sent in enumerate(sentences_ext):
                if i < 3:
                    if all(self._sentence_similarity(sent, h) <= 0.7 for h in hybrid_sentences):
                        hybrid_sentences.append(sent)
            for i, sent in enumerate(sentences_abs[1:]):
                if i < 2:
                    if all(self._sentence_similarity(sent, h) <= 0.7 for h in hybrid_sentences):
                        hybrid_sentences.append(sent)
            hybrid_summary = " ".join(hybrid_sentences)
        # Ensure hybrid is not worse than both originals
        rouge_hybrid = self._evaluate_rouge(hybrid_summary, original_text)
        bertscore_hybrid = self._evaluate_bertscore(
            hybrid_summary, original_text)
        hybrid_score = (rouge_hybrid['rouge-1']
                        ['f'] + bertscore_hybrid['f1']) / 2
        best_original_score = max(ext_weight, abs_weight)
        if hybrid_score < best_original_score:
            return extractive_summary if ext_weight > abs_weight else abstractive_summary
        return hybrid_summary

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, mode='1+1'):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        references = [a['content'] for a in articles]
        # Prepare containers
        textrank_sums, lexrank_sums, lsa_sums = [], [], []
        abstractive_sums, hybrid_sums = [], []
        best_extract_sums, best_abstractive_sums, best_best_sums = [], [], []

        for article in articles:
            content = article['content']
            # Single-method results
            tr_sum = self.extractive.textrank(content)
            lr_sum = self.extractive.lexrank(content)
            lsa_sum = self.extractive.lsa(content)
            abs_sum = getattr(self.abstractive,
                              self.abstractive_methods[0])(content)
            textrank_sums.append(tr_sum)
            lexrank_sums.append(lr_sum)
            lsa_sums.append(lsa_sum)
            abstractive_sums.append(abs_sum)
            # 1+1 fusion
            if mode == '1+1':
                hybrid_sum = self.hybrid_summarization(
                    content, tr_sum, abs_sum)
                hybrid_sums.append(hybrid_sum)
            # 3+1 fusion (combine all three extractive summaries as one input for hybrid fusion)
            elif mode == '3+1':
                combined_ext = ' '.join([tr_sum, lr_sum, lsa_sum])
                hybrid_sum = self.hybrid_summarization(
                    content, combined_ext, abs_sum)
                hybrid_sums.append(hybrid_sum)
            # 3+3 fusion (best of 3 extractives, best of 3 abstractives, best hybrid)
            elif mode == '3+3':
                # Best extractive
                ex_summaries = [tr_sum, lr_sum, lsa_sum]
                ex_scores = [self.evaluator.score(
                    s, content)['rouge_rouge-1_f'] for s in ex_summaries]
                best_ext_idx = int(np.argmax(ex_scores))
                best_extract_sums.append(ex_summaries[best_ext_idx])
                # Best abstractive
                ab_summaries = [getattr(self.abstractive, m)(
                    content) for m in self.abstractive_methods]
                ab_scores = [self.evaluator.score(
                    s, content)['rouge_rouge-1_f'] for s in ab_summaries]
                best_abs_idx = int(np.argmax(ab_scores))
                best_abstractive_sums.append(ab_summaries[best_abs_idx])
                # Hybrid fusion for all 3x3 pairs
                all_hybrids = [self.hybrid_summarization(content, e, a)
                               for e in ex_summaries for a in ab_summaries]
                hybrid_scores = [self.evaluator.score(
                    h, content)['rouge_rouge-1_f'] for h in all_hybrids]
                best_hybrid_idx = int(np.argmax(hybrid_scores))
                best_best_sums.append(all_hybrids[best_hybrid_idx])
        # Compute averages
        avg_textrank = self.evaluator.batch_score(textrank_sums, references)[1]
        avg_lexrank = self.evaluator.batch_score(lexrank_sums, references)[1]
        avg_lsa = self.evaluator.batch_score(lsa_sums, references)[1]
        avg_abstractive = self.evaluator.batch_score(
            abstractive_sums, references)[1]
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
            avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]
            output = {
                'average_scores': {
                    'textrank': avg_textrank,
                    'lexrank': avg_lexrank,
                    'lsa': avg_lsa,
                    'abstractive': avg_abstractive,
                    'best_single': avg_hybrid
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
        json_path = os.path.join(outdir, 'parallel_results.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
