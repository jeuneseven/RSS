"""
pipelines/pipeline_serial.py (compare single + hybrid)

Serial pipeline: outputs scores for single extractive, single abstractive, and hybrid summary.
Generates a bar plot to compare all three methods.
"""

import os
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json


class SerialPipeline:
    def __init__(self, extractive_method='textrank', abstractive_method='bart',
                 max_length=150, min_length=50, num_beams=4, device=None):
        self.extractive_method = extractive_method
        self.abstractive_method = abstractive_method
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, combine=False):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]
        # To store all summaries for evaluation
        ex_summaries, ab_summaries, hy_summaries = [], [], []

        for i, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            # --- Single Extractive ---
            ext_summary = getattr(self.extractive, self.extractive_method)(
                article['content'])
            item['extractive_summary'] = ext_summary
            # --- Single Abstractive ---
            abs_summary = getattr(self.abstractive, self.abstractive_method)(
                article['content'])
            item['abstractive_summary'] = abs_summary
            # --- Hybrid (pipeline: extractive as prompt for abstractive) ---
            if combine:
                sum_tr = self.extractive.textrank(article['content'])
                sum_lr = self.extractive.lexrank(article['content'])
                sum_lsa = self.extractive.lsa(article['content'])
                hybrid_prompt = ' '.join([sum_tr, sum_lr, sum_lsa])
            else:
                hybrid_prompt = ext_summary
            hybrid_summary = getattr(
                self.abstractive, self.abstractive_method)(hybrid_prompt)
            item['hybrid_summary'] = hybrid_summary
            # --- Score each summary against original ---
            item['extractive_scores'] = self.evaluator.score(
                ext_summary, article['content'])
            item['abstractive_scores'] = self.evaluator.score(
                abs_summary, article['content'])
            item['hybrid_scores'] = self.evaluator.score(
                hybrid_summary, article['content'])
            ex_summaries.append(ext_summary)
            ab_summaries.append(abs_summary)
            hy_summaries.append(hybrid_summary)
            results_json.append(item)

        # Batch/average evaluation
        avg_ex = self.evaluator.batch_score(ex_summaries, references)[1]
        avg_ab = self.evaluator.batch_score(ab_summaries, references)[1]
        avg_hy = self.evaluator.batch_score(hy_summaries, references)[1]
        output = {
            'articles': results_json,
            'average_scores': {
                'extractive': avg_ex,
                'abstractive': avg_ab,
                'hybrid': avg_hy
            }
        }
        # Save JSON
        json_path = os.path.join(
            outdir, f'serial_{self.extractive_method}_{self.abstractive_method}.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
