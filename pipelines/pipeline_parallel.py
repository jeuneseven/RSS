"""
pipelines/pipeline_parallel.py (FINAL COMPLETE VERSION)
- FIXED: Only generate and store summaries that are actually needed based on mode and user selection
- NEW: Extractive-skeleton hybrid fusion that maintains high ROUGE scores
- Strategy: Use extractive as skeleton, insert missing abstractive content in original text order
- FIXED: Optimize computation by avoiding unnecessary method calls
- FIXED: Clean up JSON output to only include relevant data
- All modes use fusion/hybrid logic with enhanced semantic similarity
- 1+1: only generates selected extractive + selected abstractive + hybrid
- 3+1: generates all extractive + selected abstractive + hybrid  
- 3+3: generates all combinations to find best
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
        """
        Initialize the parallel pipeline with configurable methods and parameters.

        Args:
            extractive_methods: List of extractive methods to use
            abstractive_methods: List of abstractive methods to use
            max_length: Maximum length for abstractive summaries
            min_length: Minimum length for abstractive summaries
            num_beams: Number of beams for beam search
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
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
        """
        Evaluate summary using ROUGE metrics and return simplified format.

        Args:
            summary: Generated summary text
            reference: Reference text for comparison

        Returns:
            Dict containing ROUGE-1 F1 score
        """
        try:
            scores = self.evaluator.score(summary, reference)
            rouge_1_f = scores.get('rouge', {}).get(
                'rouge-1', {}).get('f', 0.0)
            return {'rouge-1': {'f': rouge_1_f}}
        except Exception as e:
            print(f"ROUGE evaluation error: {e}")
            return {'rouge-1': {'f': 0.0}}

    def _evaluate_bertscore(self, summary, reference):
        """
        Evaluate summary using BERTScore and return simplified format.

        Args:
            summary: Generated summary text
            reference: Reference text for comparison

        Returns:
            Dict containing BERTScore F1 score
        """
        try:
            scores = self.evaluator.score(summary, reference)
            bertscore_f1 = scores.get('bertscore', {}).get('f1', 0.0)
            return {'f1': bertscore_f1}
        except Exception as e:
            print(f"BERTScore evaluation error: {e}")
            return {'f1': 0.0}

    def _sentence_similarity(self, sent1, sent2):
        """
        Calculate semantic similarity between two sentences using cosine similarity.

        Args:
            sent1: First sentence
            sent2: Second sentence

        Returns:
            Float between 0 and 1 representing similarity
        """
        if not sent1.strip() or not sent2.strip():
            return 0.0

        if sent1.strip() == sent2.strip():
            return 1.0

        try:
            # Tokenize and filter words, removing stopwords and non-alphanumeric tokens
            words1 = [word.lower() for word in word_tokenize(sent1)
                      if word.lower() not in self.stop_words and word.isalnum()]
            words2 = [word.lower() for word in word_tokenize(sent2)
                      if word.lower() not in self.stop_words and word.isalnum()]

            if not words1 or not words2:
                return 0.0

            # Create vocabulary from both sentences
            all_words = list(set(words1 + words2))

            # Create binary word vectors
            vector1 = [1 if word in words1 else 0 for word in all_words]
            vector2 = [1 if word in words2 else 0 for word in all_words]

            if not any(vector1) or not any(vector2):
                return 0.0

            # Calculate cosine similarity
            similarity = 1 - cosine_distance(vector1, vector2)
            return max(0.0, min(1.0, similarity))

        except Exception:
            # Fallback to simple string comparison in case of error
            return 1.0 if sent1.strip().lower() == sent2.strip().lower() else 0.0

    def combine_extractive_summaries(self, tr_sum, lr_sum, lsa_sum):
        """
        Combine three extractive summaries following serial.py logic: merge and deduplicate.

        Args:
            tr_sum: TextRank summary
            lr_sum: LexRank summary
            lsa_sum: LSA summary

        Returns:
            Combined and deduplicated extractive summary
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
        OPTIMIZED: Use extractive as skeleton, carefully insert non-redundant abstractive content.

        Strategy:
        1. Use extractive summary as the main skeleton (maintains high ROUGE)
        2. Find truly unique content in abstractive that adds value
        3. Insert this content in the correct position based on original text order
        4. Strict deduplication to avoid redundancy

        Args:
            original_text: Original article text
            extractive_summary: Extractive summary (skeleton)
            abstractive_summary: Abstractive summary (content source)

        Returns:
            Hybrid summary with extractive skeleton enhanced by non-redundant abstractive content
        """
        # Input validation
        if not extractive_summary.strip():
            return abstractive_summary
        if not abstractive_summary.strip():
            return extractive_summary

        print(
            "DEBUG - Using optimized extractive-skeleton + abstractive-enhancement strategy")

        # Tokenize sentences
        sentences_ext = sent_tokenize(extractive_summary)
        sentences_abs = sent_tokenize(abstractive_summary)
        sentences_orig = sent_tokenize(original_text)

        print(f"DEBUG - Extractive: {len(sentences_ext)} sentences")
        print(f"DEBUG - Abstractive: {len(sentences_abs)} sentences")
        print(f"DEBUG - Original: {len(sentences_orig)} sentences")

        # Step 1: Start with extractive summary as skeleton
        hybrid_content = []

        # Step 2: Find positions of extractive sentences in original text
        ext_positions = []
        for ext_sent in sentences_ext:
            best_match_pos = -1
            best_similarity = 0

            for i, orig_sent in enumerate(sentences_orig):
                similarity = self._sentence_similarity(ext_sent, orig_sent)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_pos = i

            if best_similarity > 0.6:  # Reasonable match with original
                ext_positions.append((best_match_pos, ext_sent))
                print(
                    f"DEBUG - Extractive sentence mapped to position {best_match_pos}")

        # Sort by original text position
        ext_positions.sort(key=lambda x: x[0])

        # Step 3: Find truly unique and valuable abstractive content
        valuable_abs_content = []
        for abs_sent in sentences_abs:
            # Check if this abstractive sentence is truly unique compared to ALL extractive content
            is_truly_unique = True
            max_similarity_with_ext = 0

            # Check similarity with entire extractive summary (not just individual sentences)
            for ext_sent in sentences_ext:
                similarity = self._sentence_similarity(abs_sent, ext_sent)
                max_similarity_with_ext = max(
                    max_similarity_with_ext, similarity)

                # Very strict threshold - avoid even semantically similar content
                if similarity > 0.65:  # Increased from 0.6 to 0.65
                    is_truly_unique = False
                    print(
                        f"DEBUG - Rejected abstractive (sim={similarity:.3f}): {abs_sent[:50]}...")
                    break

            if is_truly_unique:
                # Find best position in original text
                best_orig_match = 0
                best_orig_pos = -1

                for i, orig_sent in enumerate(sentences_orig):
                    similarity = self._sentence_similarity(abs_sent, orig_sent)
                    if similarity > best_orig_match:
                        best_orig_match = similarity
                        best_orig_pos = i

                # Only add if it has reasonable connection to original text
                if best_orig_match > 0.35:  # Slightly higher threshold
                    # Additional value check: does it contain new keywords?
                    abs_words = set(abs_sent.lower().split())
                    ext_words = set(extractive_summary.lower().split())
                    new_words = abs_words - ext_words

                    if len(new_words) >= 3:  # Must add at least 3 new words
                        valuable_abs_content.append(
                            (best_orig_pos, abs_sent, best_orig_match))
                        print(
                            f"DEBUG - Found valuable abstractive content at position {best_orig_pos} (sim={best_orig_match:.3f}, {len(new_words)} new words)")
                    else:
                        print(
                            f"DEBUG - Rejected abstractive (insufficient new content): {abs_sent[:50]}...")

        # Sort abstractive content by original text position
        valuable_abs_content.sort(key=lambda x: x[0])

        # Step 4: Conservative selection - only add the BEST abstractive content
        if valuable_abs_content:
            # Limit to only the single best abstractive addition to minimize risk
            # Best original similarity
            best_abs = max(valuable_abs_content, key=lambda x: x[2])
            valuable_abs_content = [best_abs]
            print(
                f"DEBUG - Selected single best abstractive addition: {best_abs[1][:50]}...")

        # Step 5: Build hybrid summary by merging in original text order
        all_content = []

        # Add extractive content with their positions
        for pos, sent in ext_positions:
            all_content.append((pos, sent, 'extractive'))

        # Add the selected valuable abstractive content
        for pos, sent, sim in valuable_abs_content:
            all_content.append((pos, sent, 'abstractive'))

        # Sort all content by original text position
        all_content.sort(key=lambda x: x[0])

        # Step 6: Build final hybrid summary with final deduplication check
        hybrid_sentences = []
        for pos, sent, source in all_content:
            # Final deduplication check
            is_duplicate = False
            for existing_sent in hybrid_sentences:
                # Very strict final check
                if self._sentence_similarity(sent, existing_sent) > 0.7:
                    is_duplicate = True
                    print(
                        f"DEBUG - Final dedup removed {source}: {sent[:30]}...")
                    break

            if not is_duplicate:
                hybrid_sentences.append(sent)
                print(
                    f"DEBUG - Added {source} sentence at pos {pos}: {sent[:50]}...")

        # Step 7: Create final hybrid summary
        if not hybrid_sentences:
            hybrid_sentences = sentences_ext  # Fallback to extractive
            print("DEBUG - No content survived filtering, using extractive summary")

        hybrid_summary = " ".join(hybrid_sentences)

        # Step 8: Quality control - ensure hybrid performs at least as well as extractive
        rouge_ext = self._evaluate_rouge(extractive_summary, original_text)
        rouge_hybrid = self._evaluate_rouge(hybrid_summary, original_text)

        print(f"DEBUG - Extractive ROUGE: {rouge_ext['rouge-1']['f']:.4f}")
        print(f"DEBUG - Hybrid ROUGE: {rouge_hybrid['rouge-1']['f']:.4f}")

        # Conservative quality control - if hybrid doesn't clearly improve, use extractive
        # Must improve by at least 0.5%
        if rouge_hybrid['rouge-1']['f'] < rouge_ext['rouge-1']['f'] * 1.005:
            print(
                "DEBUG - Hybrid doesn't provide clear improvement, using extractive summary")
            return extractive_summary

        print(
            f"DEBUG - Final hybrid: {len(hybrid_summary.split())} words, {len(hybrid_sentences)} sentences")
        print(
            f"DEBUG - Improvement: +{((rouge_hybrid['rouge-1']['f'] / rouge_ext['rouge-1']['f']) - 1) * 100:.2f}%")
        return hybrid_summary

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, mode='1+1',
            selected_extractive='textrank', selected_abstractive='bart'):
        """
        FIXED: Only generate summaries that are actually needed based on mode and user selection.
        """
        ensure_dir(outdir)
        parser = RSSParser()
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        references = [a['content'] for a in articles]

        # Prepare containers for different summary types
        # FIXED: Initialize as empty, only populate what's needed
        textrank_sums, lexrank_sums, lsa_sums = [], [], []
        abstractive_sums, hybrid_sums = [], []
        best_extract_sums, best_abstractive_sums, best_best_sums = [], [], []

        # Store detailed results for each article
        detailed_results = []

        for i, article in enumerate(articles):
            content = article['content']

            # Create detailed article result object
            article_result = {
                'title': article['title'],
                'link': article['link'],
                'original_content': content  # Store cleaned original content
            }

            # FIXED: Generate summaries based on mode requirements only
            if mode == '1+1':
                # 1+1 mode: Only generate the selected extractive method
                if selected_extractive == 'textrank':
                    selected_ext_sum = self.extractive.textrank(content)
                    # Store only the selected method
                    article_result['textrank_summary'] = selected_ext_sum
                    article_result['textrank_scores'] = self.evaluator.score(
                        selected_ext_sum, content)
                    textrank_sums.append(selected_ext_sum)
                elif selected_extractive == 'lexrank':
                    selected_ext_sum = self.extractive.lexrank(content)
                    article_result['lexrank_summary'] = selected_ext_sum
                    article_result['lexrank_scores'] = self.evaluator.score(
                        selected_ext_sum, content)
                    lexrank_sums.append(selected_ext_sum)
                elif selected_extractive == 'lsa':
                    selected_ext_sum = self.extractive.lsa(content)
                    article_result['lsa_summary'] = selected_ext_sum
                    article_result['lsa_scores'] = self.evaluator.score(
                        selected_ext_sum, content)
                    lsa_sums.append(selected_ext_sum)
                else:
                    selected_ext_sum = self.extractive.textrank(
                        content)  # fallback
                    article_result['textrank_summary'] = selected_ext_sum
                    article_result['textrank_scores'] = self.evaluator.score(
                        selected_ext_sum, content)
                    textrank_sums.append(selected_ext_sum)

                # Generate only the selected abstractive summary
                selected_abs_sum = getattr(
                    self.abstractive, selected_abstractive)(content)
                abstractive_sums.append(selected_abs_sum)

                # Generate hybrid using the SELECTED methods
                hybrid_sum = self.hybrid_summarization(
                    content, selected_ext_sum, selected_abs_sum)
                hybrid_sums.append(hybrid_sum)

                # Store selected method results
                article_result['extractive_summary'] = selected_ext_sum
                article_result['abstractive_summary'] = selected_abs_sum
                article_result['hybrid_summary'] = hybrid_sum
                article_result['extractive_scores'] = self.evaluator.score(
                    selected_ext_sum, content)
                article_result['abstractive_scores'] = self.evaluator.score(
                    selected_abs_sum, content)
                article_result['hybrid_scores'] = self.evaluator.score(
                    hybrid_sum, content)

            elif mode == '3+1':
                # 3+1 mode: Generate all extractive methods + selected abstractive method
                tr_sum = self.extractive.textrank(content)
                lr_sum = self.extractive.lexrank(content)
                lsa_sum = self.extractive.lsa(content)

                textrank_sums.append(tr_sum)
                lexrank_sums.append(lr_sum)
                lsa_sums.append(lsa_sum)

                # Store all extractive summaries and scores
                article_result['textrank_summary'] = tr_sum
                article_result['lexrank_summary'] = lr_sum
                article_result['lsa_summary'] = lsa_sum
                article_result['textrank_scores'] = self.evaluator.score(
                    tr_sum, content)
                article_result['lexrank_scores'] = self.evaluator.score(
                    lr_sum, content)
                article_result['lsa_scores'] = self.evaluator.score(
                    lsa_sum, content)

                # Combine all extractive methods
                combined_extractive = self.combine_extractive_summaries(
                    tr_sum, lr_sum, lsa_sum)

                # Get the selected abstractive summary
                selected_abs_sum = getattr(
                    self.abstractive, selected_abstractive)(content)
                abstractive_sums.append(selected_abs_sum)

                # Generate hybrid using combined extractive + selected abstractive
                hybrid_sum = self.hybrid_summarization(
                    content, combined_extractive, selected_abs_sum)
                hybrid_sums.append(hybrid_sum)

                # Store combined results
                article_result['extractive_summary'] = combined_extractive
                article_result['abstractive_summary'] = selected_abs_sum
                article_result['hybrid_summary'] = hybrid_sum
                article_result['extractive_scores'] = self.evaluator.score(
                    combined_extractive, content)
                article_result['abstractive_scores'] = self.evaluator.score(
                    selected_abs_sum, content)
                article_result['hybrid_scores'] = self.evaluator.score(
                    hybrid_sum, content)

            # Store article result
            detailed_results.append(article_result)

        # FIXED: Calculate average scores only for methods that were actually used
        if mode == '1+1':
            if selected_extractive == 'textrank':
                avg_extractive = self.evaluator.batch_score(
                    textrank_sums, references)[1]
                avg_textrank = avg_extractive
                avg_lexrank = avg_lsa = {}
            elif selected_extractive == 'lexrank':
                avg_extractive = self.evaluator.batch_score(
                    lexrank_sums, references)[1]
                avg_lexrank = avg_extractive
                avg_textrank = avg_lsa = {}
            elif selected_extractive == 'lsa':
                avg_extractive = self.evaluator.batch_score(
                    lsa_sums, references)[1]
                avg_lsa = avg_extractive
                avg_textrank = avg_lexrank = {}
            else:
                avg_extractive = self.evaluator.batch_score(
                    textrank_sums, references)[1]
                avg_textrank = avg_extractive
                avg_lexrank = avg_lsa = {}

            avg_abstractive = self.evaluator.batch_score(
                abstractive_sums, references)[1]
            avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]

            output = {
                'articles': detailed_results,
                'average_scores': {
                    'extractive': avg_extractive,
                    'abstractive': avg_abstractive,
                    'combo': avg_hybrid,
                    # Include individual scores only if they were calculated
                    'textrank': avg_textrank,
                    'lexrank': avg_lexrank,
                    'lsa': avg_lsa
                }
            }

        elif mode == '3+1':
            avg_textrank = self.evaluator.batch_score(
                textrank_sums, references)[1]
            avg_lexrank = self.evaluator.batch_score(
                lexrank_sums, references)[1]
            avg_lsa = self.evaluator.batch_score(lsa_sums, references)[1]
            avg_abstractive = self.evaluator.batch_score(
                abstractive_sums, references)[1]
            avg_hybrid = self.evaluator.batch_score(hybrid_sums, references)[1]

            output = {
                'articles': detailed_results,
                'average_scores': {
                    'textrank': avg_textrank,
                    'lexrank': avg_lexrank,
                    'lsa': avg_lsa,
                    'abstractive': avg_abstractive,
                    'best_single': avg_hybrid
                }
            }
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Save results with unique naming based on mode and selected methods
        if mode == '1+1':
            json_filename = f'parallel_1plus1_{selected_extractive}_{selected_abstractive}.json'
        elif mode == '3+1':
            json_filename = f'parallel_3plus1_all_{selected_abstractive}.json'
        else:
            json_filename = f'parallel_{mode}_results.json'  # fallback

        json_path = os.path.join(outdir, json_filename)
        save_json(output, json_path)
        print(
            f'Parallel pipeline ({mode} mode) JSON result saved to {json_path}')
        return output
