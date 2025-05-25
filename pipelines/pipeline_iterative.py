"""
pipeline_iterative.py (final output: bar groups = all single-methods on original + iterative refinement result)
- Five bar groups: textrank(original), lexrank(original), lsa(original), bart(original), hybrid(iterative refinement)
- 'hybrid' is the true iterative-refined summary, others are direct single-methods
"""

import os
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from evaluation.scorer import Evaluator
from utils.rss_parser import RSSParser
from utils.common import ensure_dir, save_json
from nltk.tokenize import sent_tokenize


class IterativePipeline:
    def __init__(self, abstractive_method='bart', max_length=150, min_length=50, num_beams=4, device=None, reduction_ratio=0.8):
        self.abstractive_method = abstractive_method
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.device = device
        self.reduction_ratio = reduction_ratio
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5):
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        references = [a['content'] for a in articles]
        # Direct methods on original text
        textrank_sums, lexrank_sums, lsa_sums, abstractive_sums = [], [], [], []
        # Iterative refinement hybrid
        hybrid_sums = []
        MIN_TOKENS = 50
        MAX_TOKENS = 150
        reduction_ratio = self.reduction_ratio
        for article in articles:
            content = article['content']
            # All single-method summaries on original text
            tr_sum = self.extractive.textrank(content)
            lr_sum = self.extractive.lexrank(content)
            lsa_sum = self.extractive.lsa(content)
            abs_sum = getattr(self.abstractive,
                              self.abstractive_method)(content)
            textrank_sums.append(tr_sum)
            lexrank_sums.append(lr_sum)
            lsa_sums.append(lsa_sum)
            abstractive_sums.append(abs_sum)
            # Iterative refinement
            # Stage 1: TextRank
            tr_target = min(MAX_TOKENS, max(
                MIN_TOKENS, int(len(content.split()) * 0.3)))
            tr_sentc = max(
                1, int(tr_target / (len(tr_sum.split()) / max(1, len(sent_tokenize(tr_sum))))))
            iter_tr = self.extractive.textrank(content, num_sentences=tr_sentc)
            # Stage 2: LexRank on TextRank output
            lr_target = max(MIN_TOKENS, int(
                len(iter_tr.split()) * reduction_ratio))
            lr_sentc = max(1, int(
                lr_target / (len(iter_tr.split()) / max(1, len(sent_tokenize(iter_tr))))))
            iter_lr = self.extractive.lexrank(iter_tr, num_sentences=lr_sentc)
            # Stage 3: LSA on LexRank output
            lsa_target = max(MIN_TOKENS, int(
                len(iter_lr.split()) * reduction_ratio))
            lsa_sentc = max(1, int(
                lsa_target / (len(iter_lr.split()) / max(1, len(sent_tokenize(iter_lr))))))
            iter_lsa = self.extractive.lsa(iter_lr, num_sentences=lsa_sentc)
            # Stage 4: Abstractive on LSA output
            abs_target = max(MIN_TOKENS, min(MAX_TOKENS, int(
                len(iter_lsa.split()) * reduction_ratio)))
            iter_abs = getattr(
                self.abstractive, self.abstractive_method)(iter_lsa)
            # Hybrid = iterative-refined result
            hybrid_sums.append(iter_abs)
        avg_textrank = self.evaluator.batch_score(textrank_sums, references)[1]
        avg_lexrank = self.evaluator.batch_score(lexrank_sums, references)[1]
        avg_lsa = self.evaluator.batch_score(lsa_sums, references)[1]
        avg_abstractive = self.evaluator.batch_score(
            abstractive_sums, references)[1]
        avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]
        output = {
            'average_scores': {
                'textrank': avg_textrank,
                'lexrank': avg_lexrank,
                'lsa': avg_lsa,
                'abstractive': avg_abstractive,
                'final': avg_hybrid
            }
        }
        json_path = os.path.join(outdir, 'iterative_results.json')
        save_json(output, json_path)
        print(f'JSON result saved to {json_path}')
        return output
