"""
pipeline_serial.py (Enhanced with intelligent extractive combination)
- Upgraded combine mode to use semantic deduplication like parallel pipeline
- Maintains compatibility with existing interface and evaluation structure
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

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class SerialPipeline:
    def __init__(self, extractive_method='textrank', abstractive_method='bart',
                 max_length=150, min_length=50, num_beams=4, device=None):
        self.extractive_method = extractive_method
        self.abstractive_method = abstractive_method
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer(
            max_length=max_length, min_length=min_length, num_beams=num_beams, device=device)
        self.evaluator = Evaluator()

        # Initialize stopwords for intelligent combination
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

    def _sentence_similarity(self, sent1, sent2):
        """
        Calculate semantic similarity between two sentences using cosine similarity.
        Enhanced from parallel pipeline for intelligent deduplication.

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
        Intelligently combine three extractive summaries with semantic deduplication.
        Enhanced version replacing simple string concatenation.

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

        # Remove empty sentences
        all_sentences = [sent.strip()
                         for sent in all_sentences if sent.strip()]

        if not all_sentences:
            return ""

        # Semantic deduplication using similarity threshold
        unique_sentences = []
        similarity_threshold = 0.8  # Sentences with >80% similarity are considered duplicates

        for sent in all_sentences:
            is_unique = True
            for existing_sent in unique_sentences:
                if self._sentence_similarity(sent, existing_sent) > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_sentences.append(sent)

        # Limit to reasonable length (7 sentences max for abstractive model input)
        max_sentences = 7
        final_sentences = unique_sentences[:max_sentences]

        return " ".join(final_sentences)

    def run(self, rss_path_or_url, outdir='data/outputs/', max_articles=5, combine=False):
        """
        Enhanced serial pipeline with proper zero-shot classification data preservation.
        Original functionality and output format remain unchanged.
        """
        ensure_dir(outdir)
        parser = RSSParser()  # Now uses zero-shot classification
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]
        ex_summaries, ab_summaries, hy_summaries = [], [], []
        tr_summaries, lr_summaries, lsa_summaries = [], [], []

        # Silent category tracking
        category_stats = {}

        for i, article in enumerate(articles):
            print(f"\n📝 Processing article {i+1} for summarization...")

            # Extract article data including ALL classification info
            item = {
                'title': article['title'],
                'link': article['link'],
                'original_content': article['content']
            }

            # IMPORTANT: Preserve ALL classification data from RSS parser
            classification_fields = [
                'category', 'classification_confidence', 'classification_scores',
                'top_3_predictions', 'classification_method', 'classification_error'
            ]

            for field in classification_fields:
                if field in article:
                    item[field] = article[field]
                    if field == 'category':
                        print(f"   📂 Category: {article[field]}")
                    elif field == 'classification_confidence':
                        print(f"   🎯 Confidence: {article[field]:.4f}")
                    elif field == 'classification_method':
                        print(f"   🔧 Method: {article[field]}")

            # Set defaults for missing classification fields
            if 'category' not in item:
                item['category'] = 'unknown'
            if 'classification_confidence' not in item:
                item['classification_confidence'] = 0.0
            if 'classification_method' not in item:
                item['classification_method'] = 'none'

            content = article['content']
            category = item['category']  # Use the preserved category

            # Original summarization logic (completely unchanged)
            print(f"   📄 Generating summaries...")
            if combine:
                print(f"   🔗 Using combined extractive approach...")
                sum_tr = self.extractive.textrank(content)
                sum_lr = self.extractive.lexrank(content)
                sum_lsa = self.extractive.lsa(content)
                tr_summaries.append(sum_tr)
                lr_summaries.append(sum_lr)
                lsa_summaries.append(sum_lsa)
                ext_summary = self.combine_extractive_summaries(
                    sum_tr, sum_lr, sum_lsa)
                print(f"   ✅ Combined extractive summary generated")
            else:
                print(
                    f"   📊 Using {self.extractive_method} extractive method...")
                sum_tr = sum_lr = sum_lsa = None
                ext_summary = getattr(
                    self.extractive, self.extractive_method)(content)
                print(f"   ✅ {self.extractive_method} summary generated")

            print(
                f"   🤖 Generating {self.abstractive_method} abstractive summary...")
            abs_summary = getattr(
                self.abstractive, self.abstractive_method)(content)
            print(f"   ✅ Abstractive summary generated")

            print(f"   🔀 Generating hybrid summary...")
            if combine:
                hybrid_prompt = ext_summary
            else:
                hybrid_prompt = ext_summary
            hybrid_summary = getattr(
                self.abstractive, self.abstractive_method)(hybrid_prompt)
            print(f"   ✅ Hybrid summary generated")

            # Original result storage and evaluation (unchanged)
            item['extractive_summary'] = ext_summary
            item['abstractive_summary'] = abs_summary
            item['hybrid_summary'] = hybrid_summary

            print(f"   📊 Calculating evaluation scores...")
            item['extractive_scores'] = self.evaluator.score(
                ext_summary, content)
            item['abstractive_scores'] = self.evaluator.score(
                abs_summary, content)
            item['hybrid_scores'] = self.evaluator.score(
                hybrid_summary, content)

            if combine:
                item['textrank_summary'] = sum_tr
                item['lexrank_summary'] = sum_lr
                item['lsa_summary'] = sum_lsa
                item['textrank_scores'] = self.evaluator.score(sum_tr, content)
                item['lexrank_scores'] = self.evaluator.score(sum_lr, content)
                item['lsa_scores'] = self.evaluator.score(sum_lsa, content)

            # Category statistics collection (with preserved classification data)
            if category not in category_stats:
                category_stats[category] = {
                    'articles': [], 'extractive': [], 'abstractive': [], 'hybrid': []
                }
            category_stats[category]['articles'].append(
                item)  # Now includes all classification data
            category_stats[category]['extractive'].append(ext_summary)
            category_stats[category]['abstractive'].append(abs_summary)
            category_stats[category]['hybrid'].append(hybrid_summary)

            ex_summaries.append(ext_summary)
            ab_summaries.append(abs_summary)
            hy_summaries.append(hybrid_summary)
            results_json.append(item)

            print(f"   ✅ Article processing complete")

        # Original batch evaluation (unchanged)
        print(f"\n📊 Calculating overall performance metrics...")
        avg_ex = self.evaluator.batch_score(ex_summaries, references)[1]
        avg_ab = self.evaluator.batch_score(ab_summaries, references)[1]
        avg_hy = self.evaluator.batch_score(hy_summaries, references)[1]
        avg_tr = self.evaluator.batch_score(tr_summaries, references)[
            1] if combine else {}
        avg_lr = self.evaluator.batch_score(lr_summaries, references)[
            1] if combine else {}
        avg_lsa = self.evaluator.batch_score(lsa_summaries, references)[
            1] if combine else {}

        # Enhanced analysis with category insights
        print(f"\n📊 Analyzing results by category...")
        category_analysis = self._analyze_by_category(
            category_stats, references)

        # Original output structure preserved, with silent enhancement
        output = {
            'articles': results_json,
            'average_scores': {
                'extractive': avg_ex,
                'abstractive': avg_ab,
                'hybrid': avg_hy,
                'textrank': avg_tr,
                'lexrank': avg_lr,
                'lsa': avg_lsa
            },
            # Silent addition - won't break existing code that reads this JSON
            'category_analysis': category_analysis
        }

        # Original file naming and saving (unchanged)
        if combine:
            json_filename = f'serial_combined_{self.abstractive_method}.json'
        else:
            json_filename = f'serial_{self.extractive_method}_{self.abstractive_method}.json'

        json_path = os.path.join(outdir, json_filename)
        save_json(output, json_path)
        print(f'Serial pipeline JSON result saved to {json_path}')

        # Silent category insights printing
        self._print_category_insights(category_analysis)

        return output

    # pipelines/pipeline_serial.py

    def _analyze_by_category(self, category_stats, references):
        """
        Enhanced category analysis with proper confidence handling for zero-shot classification.
        Provides insights without affecting main pipeline flow.
        """
        analysis = {}

        for category, data in category_stats.items():
            if len(data['articles']) == 0:
                continue

            print(
                f"📊 Analyzing category: {category} ({len(data['articles'])} articles)")

            # Calculate classification confidence statistics with proper handling
            confidences = []
            classification_details = []
            methods_used = []

            for item in data['articles']:
                # Handle confidence values safely
                conf = item.get('classification_confidence', 0.0)
                try:
                    conf_float = float(conf) if conf is not None else 0.0
                    confidences.append(conf_float)
                except (ValueError, TypeError):
                    print(f"   ⚠️ Invalid confidence value: {conf}")
                    confidences.append(0.0)
                    conf_float = 0.0

                # Track classification methods
                method = item.get('classification_method', 'unknown')
                methods_used.append(method)

                # Build classification details
                detail_item = {
                    'title': item['title'][:50] + '...' if len(item['title']) > 50 else item['title'],
                    'confidence': conf_float,
                    'method': method
                }

                # Add top predictions if available
                if 'top_3_predictions' in item and item['top_3_predictions']:
                    detail_item['top_3'] = item['top_3_predictions']

                # Add all scores if available
                if 'classification_scores' in item and item['classification_scores']:
                    detail_item['all_scores'] = item['classification_scores']

                # Add any classification errors
                if 'classification_error' in item:
                    detail_item['error'] = item['classification_error']

                classification_details.append(detail_item)

            # Calculate classification quality metrics
            if confidences and len(confidences) > 0:
                avg_confidence = sum(confidences) / len(confidences)
                min_confidence = min(confidences)
                max_confidence = max(confidences)

                # Calculate standard deviation safely
                if len(confidences) > 1:
                    import numpy as np
                    confidence_std = float(np.std(confidences))
                else:
                    confidence_std = 0.0
            else:
                avg_confidence = min_confidence = max_confidence = confidence_std = 0.0

            # Count classification methods
            method_counts = {}
            for method in methods_used:
                method_counts[method] = method_counts.get(method, 0) + 1

            print(
                f"   🎯 Classification confidence: avg={avg_confidence:.4f}, range=[{min_confidence:.4f}, {max_confidence:.4f}]")
            print(f"   🔧 Methods used: {method_counts}")

            # Calculate summarization scores
            cat_references = [item['original_content']
                              for item in data['articles']]

            try:
                _, cat_ex_scores = self.evaluator.batch_score(
                    data['extractive'], cat_references)
                _, cat_ab_scores = self.evaluator.batch_score(
                    data['abstractive'], cat_references)
                _, cat_hy_scores = self.evaluator.batch_score(
                    data['hybrid'], cat_references)

                print(f"   📈 ROUGE-1 F1: extractive={cat_ex_scores.get('rouge_rouge-1_f', 0):.4f}, "
                      f"abstractive={cat_ab_scores.get('rouge_rouge-1_f', 0):.4f}, "
                      f"hybrid={cat_hy_scores.get('rouge_rouge-1_f', 0):.4f}")

                analysis[category] = {
                    'article_count': len(data['articles']),
                    'extractive_scores': cat_ex_scores,
                    'abstractive_scores': cat_ab_scores,
                    'hybrid_scores': cat_hy_scores,
                    'sample_titles': [item['title'] for item in data['articles'][:3]],
                    'classification_quality': {
                        'avg_confidence': avg_confidence,
                        'min_confidence': min_confidence,
                        'max_confidence': max_confidence,
                        'confidence_std': confidence_std,
                        'method_counts': method_counts
                    },
                    'classification_details': classification_details
                }
            except Exception as e:
                print(f"   ❌ Error analyzing category {category}: {e}")
                analysis[category] = {
                    'article_count': len(data['articles']),
                    'classification_quality': {
                        'avg_confidence': avg_confidence,
                        'min_confidence': min_confidence,
                        'max_confidence': max_confidence,
                        'confidence_std': confidence_std,
                        'method_counts': method_counts
                    },
                    'classification_details': classification_details,
                    'error': str(e)
                }

        return analysis

    def _print_category_insights(self, category_analysis):
        """
        Enhanced insights printing with zero-shot classification confidence details.
        Provides immediate value without changing output files.
        """
        print("\n" + "="*80)
        print("📊 CONTENT CLASSIFICATION & SUMMARIZATION ANALYSIS")
        print("="*80)

        total_articles = sum(data.get('article_count', 0)
                             for data in category_analysis.values())
        print(f"📈 Total articles processed: {total_articles}")

        # Overall classification method summary
        all_methods = {}
        for data in category_analysis.values():
            if 'classification_quality' in data and 'method_counts' in data['classification_quality']:
                for method, count in data['classification_quality']['method_counts'].items():
                    all_methods[method] = all_methods.get(method, 0) + count

        if all_methods:
            print(f"🔧 Classification methods used: {all_methods}")

        for category, data in category_analysis.items():
            if 'error' in data and 'article_count' not in data:
                continue

            count = data['article_count']
            percentage = (count / total_articles *
                          100) if total_articles > 0 else 0

            print(
                f"\n📂 {category.upper()}: {count} articles ({percentage:.1f}%)")

            # Classification confidence analysis
            if 'classification_quality' in data:
                qual = data['classification_quality']
                avg_conf = qual.get('avg_confidence', 0)
                min_conf = qual.get('min_confidence', 0)
                max_conf = qual.get('max_confidence', 0)
                std_conf = qual.get('confidence_std', 0)
                method_counts = qual.get('method_counts', {})

                print(f"   🎯 Classification Confidence:")
                print(f"      Average: {avg_conf:.4f}")
                print(f"      Range: {min_conf:.4f} - {max_conf:.4f}")
                print(f"      Std Dev: {std_conf:.4f}")
                print(f"      Methods: {method_counts}")

                # Confidence quality indicator with zero-shot specific thresholds
                if avg_conf > 0.8:
                    quality_indicator = "🟢 High confidence"
                elif avg_conf > 0.6:
                    quality_indicator = "🟡 Medium confidence"
                elif avg_conf > 0.4:
                    quality_indicator = "🟠 Low confidence"
                elif avg_conf > 0.0:
                    quality_indicator = "🔴 Very low confidence (review needed)"
                else:
                    quality_indicator = "⚫ No classification data"
                print(f"      Quality: {quality_indicator}")

            # Show low confidence items for debugging
            if 'classification_details' in data:
                low_conf_items = [item for item in data['classification_details']
                                  if item.get('confidence', 0) < 0.6]
                if low_conf_items:
                    print(
                        f"   ⚠️  Low confidence items ({len(low_conf_items)}):")
                    for item in low_conf_items[:2]:  # Show top 2
                        conf = item.get('confidence', 0)
                        method = item.get('method', 'unknown')
                        print(
                            f"      • {item['title']} (conf: {conf:.3f}, method: {method})")

                        # Show alternative predictions if available
                        if 'top_3' in item and len(item['top_3']) >= 2:
                            alt_cat, alt_score = item['top_3'][1]
                            print(
                                f"        Alternative: {alt_cat} ({alt_score:.3f})")

                        # Show classification errors if any
                        if 'error' in item:
                            print(f"        Error: {item['error']}")

            # Summarization performance
            if 'extractive_scores' in data:
                ex_rouge = data['extractive_scores'].get('rouge_rouge-1_f', 0)
                ab_rouge = data['abstractive_scores'].get('rouge_rouge-1_f', 0)
                hy_rouge = data['hybrid_scores'].get('rouge_rouge-1_f', 0)

                print(f"   📈 Summarization Performance (ROUGE-1 F1):")
                print(f"      Extractive:  {ex_rouge:.4f}")
                print(f"      Abstractive: {ab_rouge:.4f}")
                print(f"      Hybrid:      {hy_rouge:.4f}")

                # Determine best method
                methods = [('Extractive', ex_rouge),
                           ('Abstractive', ab_rouge), ('Hybrid', hy_rouge)]
                best_method, best_score = max(methods, key=lambda x: x[1])
                print(f"      🏆 Best: {best_method} ({best_score:.4f})")

                # Category-specific insights
                if category == 'technology' and ex_rouge > ab_rouge:
                    print(
                        f"      💡 Insight: Technology articles work better with extractive summarization")
                elif category == 'business' and ab_rouge > ex_rouge:
                    print(
                        f"      💡 Insight: Business articles work better with abstractive summarization")

            # Show sample titles
            if 'sample_titles' in data and data['sample_titles']:
                print(f"   📄 Sample articles:")
                for title in data['sample_titles']:
                    title_display = title[:70] + \
                        '...' if len(title) > 70 else title
                    print(f"      • {title_display}")

        print("="*80)
