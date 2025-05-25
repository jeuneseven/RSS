"""
pipeline_serial.py (fix: always fill textrank/lexrank/lsa scores in average_scores if combine=True)
- This ensures JSON output is always complete for plotting, even if combine mode
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
        ex_summaries, ab_summaries, hy_summaries = [], [], []
        tr_summaries, lr_summaries, lsa_summaries = [], [], []
        for i, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            content = article['content']
            if combine:
                # Always run all extractive methods if combine
                sum_tr = self.extractive.textrank(content)
                sum_lr = self.extractive.lexrank(content)
                sum_lsa = self.extractive.lsa(content)
                tr_summaries.append(sum_tr)
                lr_summaries.append(sum_lr)
                lsa_summaries.append(sum_lsa)
                ext_summary = sum_tr  # fallback for score only
            else:
                sum_tr = sum_lr = sum_lsa = None
                ext_summary = getattr(
                    self.extractive, self.extractive_method)(content)
            abs_summary = getattr(
                self.abstractive, self.abstractive_method)(content)
            # Hybrid
            if combine:
                hybrid_prompt = ' '.join([self.extractive.textrank(content),
                                          self.extractive.lexrank(content),
                                          self.extractive.lsa(content)])
            else:
                hybrid_prompt = ext_summary
            hybrid_summary = getattr(
                self.abstractive, self.abstractive_method)(hybrid_prompt)
            # Evaluation
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
                # Also store each extractive's score for plotting
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
        # Batch/average evaluation
        avg_ex = self.evaluator.batch_score(ex_summaries, references)[1]
        avg_ab = self.evaluator.batch_score(ab_summaries, references)[1]
        avg_hy = self.evaluator.batch_score(hy_summaries, references)[1]
        avg_tr = self.evaluator.batch_score(tr_summaries, references)[
            1] if combine else {}
        avg_lr = self.evaluator.batch_score(lr_summaries, references)[
            1] if combine else {}
        avg_lsa = self.evaluator.batch_score(lsa_summaries, references)[
            1] if combine else {}
        output = {
            'articles': results_json,
            'average_scores': {
                'extractive': avg_ex,
                'abstractive': avg_ab,
                'hybrid': avg_hy,
                'textrank': avg_tr,
                'lexrank': avg_lr,
                'lsa': avg_lsa
            }
        }
        json_path = os.path.join(
            outdir, f'serial_{self.extractive_method}_{self.abstractive_method}.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
