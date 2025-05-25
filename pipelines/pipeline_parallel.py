"""
pipelines/pipeline_parallel.py

Parallel (side-by-side) summarization:
- Supports:
  1. Single Extractive Method Combine with Single Abstractive Method
  2. Best Extractive Method Combine with Single Abstractive Method
  3. Best Extractive Method Combine with Best Abstractive Method
- All three outputs are compared to original and to each other
- Unified config, reusable models, unified output format
"""

import os
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json


class ParallelPipeline:
    def __init__(self, extractive_methods=None, abstractive_methods=None,
                 max_length=150, min_length=50, num_beams=4, device=None):
        if extractive_methods is None:
            extractive_methods = ['textrank', 'lexrank', 'lsa']
        if abstractive_methods is None:
            abstractive_methods = ['bart', 't5', 'pegasus']
        self.extractive_methods = extractive_methods
        self.abstractive_methods = abstractive_methods
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, mode='single-single'):
        """
        mode: 'single-single', 'best-single', 'best-best'
        """
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]

        # To store all outputs for averaging
        combo_summaries, best_single_summaries, best_best_summaries = [], [], []

        for i, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            content = article['content']
            # -- All extractive and abstractive --
            extractive_summaries = {}
            abstractive_summaries = {}
            for method in self.extractive_methods:
                summary = getattr(self.extractive, method)(content)
                extractive_summaries[method] = summary
            for method in self.abstractive_methods:
                summary = getattr(self.abstractive, method)(content)
                abstractive_summaries[method] = summary

            # 1. Single Extractive + Single Abstractive
            ex = extractive_summaries[self.extractive_methods[0]]
            ab = abstractive_summaries[self.abstractive_methods[0]]
            combo = ex + " " + ab
            combo_score = self.evaluator.score(combo, content)
            combo_summaries.append(combo)

            # 2. Best Extractive + Single Abstractive
            best_ex, best_ex_score = None, -float('inf')
            for k, v in extractive_summaries.items():
                score = self.evaluator.score(
                    v, content)['rouge']['rouge-1']['f']
                if score > best_ex_score:
                    best_ex = v
                    best_ex_score = score
            best_single = best_ex + " " + ab
            best_single_score = self.evaluator.score(best_single, content)
            best_single_summaries.append(best_single)

            # 3. Best Extractive + Best Abstractive
            best_ab, best_ab_score = None, -float('inf')
            for k, v in abstractive_summaries.items():
                score = self.evaluator.score(
                    v, content)['rouge']['rouge-1']['f']
                if score > best_ab_score:
                    best_ab = v
                    best_ab_score = score
            best_best = best_ex + " " + best_ab
            best_best_score = self.evaluator.score(best_best, content)
            best_best_summaries.append(best_best)

            item['extractive_summaries'] = extractive_summaries
            item['abstractive_summaries'] = abstractive_summaries
            item['combo'] = {'summary': combo, 'score': combo_score}
            item['best_single'] = {
                'summary': best_single, 'score': best_single_score}
            item['best_best'] = {
                'summary': best_best, 'score': best_best_score}
            results_json.append(item)

        # Average scores
        avg_combo = self.evaluator.batch_score(combo_summaries, references)[1]
        avg_best_single = self.evaluator.batch_score(
            best_single_summaries, references)[1]
        avg_best_best = self.evaluator.batch_score(
            best_best_summaries, references)[1]

        output = {
            'articles': results_json,
            'average_scores': {
                'combo': avg_combo,
                'best_single': avg_best_single,
                'best_best': avg_best_best
            }
        }
        # Save JSON
        json_path = os.path.join(outdir, 'parallel_comparison.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
