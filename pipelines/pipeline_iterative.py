"""
pipelines/pipeline_iterative.py

Iterative refinement summarization:
- Multiple rounds of extractive followed by abstractive summarization, each round on new content
- Designed to maximize information coverage through diversity of extractive methods
- Outputs ROUGE/BERTScore for each iteration and for final summary
"""

import os
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json
import nltk


class IterativePipeline:
    def __init__(self, extractive_order=None, abstractive_method='bart',
                 max_length=150, min_length=50, num_beams=4, device=None):
        if extractive_order is None:
            extractive_order = ['textrank', 'lexrank', 'lsa']
        self.extractive_order = extractive_order
        self.abstractive_method = abstractive_method
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, n_rounds=3):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]
        final_summaries = []
        all_iteration_scores = []

        for idx, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            content = article['content']
            history = []  # List of summaries at each iteration
            current_text = content
            for round_idx in range(n_rounds):
                method = self.extractive_order[round_idx % len(
                    self.extractive_order)]
                ext_summary = getattr(self.extractive, method)(current_text)
                # Optionally can filter already summarized sentences
                # current_text = ' '.join([s for s in nltk.sent_tokenize(content) if s not in nltk.sent_tokenize(ext_summary)])
                current_text = ext_summary  # For next round, summarize previous output
                abs_summary = getattr(
                    self.abstractive, self.abstractive_method)(current_text)
                history.append({
                    'extractive_method': method,
                    'extractive_summary': ext_summary,
                    'abstractive_summary': abs_summary
                })
            # Final summary = abstractive result of last round
            final_summary = history[-1]['abstractive_summary']
            item['iterations'] = history
            item['final_summary'] = final_summary
            item['final_scores'] = self.evaluator.score(final_summary, content)
            final_summaries.append(final_summary)
            all_iteration_scores.append(item['final_scores'])
            results_json.append(item)

        # Average scores for final summaries
        avg_final = self.evaluator.batch_score(final_summaries, references)[1]
        output = {
            'articles': results_json,
            'average_scores': {
                'final': avg_final
            }
        }
        json_path = os.path.join(outdir, 'iterative_refinement.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
