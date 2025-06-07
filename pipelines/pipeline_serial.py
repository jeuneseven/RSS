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
        Enhanced serial pipeline with silent classification analysis.
        Original functionality and output format remain unchanged.
        """
        ensure_dir(outdir)
        parser = RSSParser()  # No interface change needed
        articles = parser.parse(rss_path_or_url, max_articles=max_articles)
        results_json = []
        references = [a['content'] for a in articles]
        ex_summaries, ab_summaries, hy_summaries = [], [], []
        tr_summaries, lr_summaries, lsa_summaries = [], [], []

        # Silent category tracking
        category_stats = {}

        for i, article in enumerate(articles):
            item = {'title': article['title'], 'link': article['link']}
            content = article['content']
            # Silent category extraction
            category = article.get('category', 'unknown')

            # Original content storage
            item['original_content'] = content

            # Original summarization logic (completely unchanged)
            if combine:
                sum_tr = self.extractive.textrank(content)
                sum_lr = self.extractive.lexrank(content)
                sum_lsa = self.extractive.lsa(content)
                tr_summaries.append(sum_tr)
                lr_summaries.append(sum_lr)
                lsa_summaries.append(sum_lsa)
                ext_summary = self.combine_extractive_summaries(
                    sum_tr, sum_lr, sum_lsa)
            else:
                sum_tr = sum_lr = sum_lsa = None
                ext_summary = getattr(
                    self.extractive, self.extractive_method)(content)

            abs_summary = getattr(
                self.abstractive, self.abstractive_method)(content)

            if combine:
                hybrid_prompt = ext_summary
            else:
                hybrid_prompt = ext_summary
            hybrid_summary = getattr(
                self.abstractive, self.abstractive_method)(hybrid_prompt)

            # Original result storage and evaluation (unchanged)
            item['extractive_summary'] = ext_summary
            item['abstractive_summary'] = abs_summary
            item['hybrid_summary'] = hybrid_summary
            item['extractive_scores'] = self.evaluator.score(
                ext_summary, content)
            item['abstractive_scores'] = self.evaluator.score(
                abs_summary, content)
            item['hybrid_scores'] = self.evaluator.score(
                hybrid_summary, content)

            # Silent category addition - doesn't break existing JSON readers
            item['category'] = category

            if combine:
                item['textrank_summary'] = sum_tr
                item['lexrank_summary'] = sum_lr
                item['lsa_summary'] = sum_lsa
                item['textrank_scores'] = self.evaluator.score(sum_tr, content)
                item['lexrank_scores'] = self.evaluator.score(sum_lr, content)
                item['lsa_scores'] = self.evaluator.score(sum_lsa, content)

            # Category statistics collection
            if category not in category_stats:
                category_stats[category] = {
                    'articles': [], 'extractive': [], 'abstractive': [], 'hybrid': []
                }
            category_stats[category]['articles'].append(item)
            category_stats[category]['extractive'].append(ext_summary)
            category_stats[category]['abstractive'].append(abs_summary)
            category_stats[category]['hybrid'].append(hybrid_summary)

            ex_summaries.append(ext_summary)
            ab_summaries.append(abs_summary)
            hy_summaries.append(hybrid_summary)
            results_json.append(item)

        # Original batch evaluation (unchanged)
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
        Enhanced category analysis including classification confidence metrics.
        """
        analysis = {}

        for category, data in category_stats.items():
            if len(data['articles']) == 0:
                continue

            # Calculate classification confidence statistics
            confidences = []
            classification_details = []

            for item in data['articles']:
                conf = item.get('classification_confidence', 0.0)
                confidences.append(conf)

                if 'top_3_predictions' in item:
                    classification_details.append({
                        'title': item['title'][:50] + '...',
                        'confidence': conf,
                        'top_3': item['top_3_predictions']
                    })

            # Calculate classification quality metrics
            avg_confidence = sum(confidences) / \
                len(confidences) if confidences else 0.0
            min_confidence = min(confidences) if confidences else 0.0
            max_confidence = max(confidences) if confidences else 0.0

            # Existing summarization analysis
            cat_references = [item['original_content']
                              for item in data['articles']]

            try:
                _, cat_ex_scores = self.evaluator.batch_score(
                    data['extractive'], cat_references)
                _, cat_ab_scores = self.evaluator.batch_score(
                    data['abstractive'], cat_references)
                _, cat_hy_scores = self.evaluator.batch_score(
                    data['hybrid'], cat_references)

                analysis[category] = {
                    'article_count': len(data['articles']),
                    'extractive_scores': cat_ex_scores,
                    'abstractive_scores': cat_ab_scores,
                    'hybrid_scores': cat_hy_scores,
                    'sample_titles': [item['title'] for item in data['articles'][:3]],
                    # New classification metrics
                    'classification_quality': {
                        'avg_confidence': avg_confidence,
                        'min_confidence': min_confidence,
                        'max_confidence': max_confidence,
                        'confidence_std': np.std(confidences) if len(confidences) > 1 else 0.0
                    },
                    'classification_details': classification_details
                }
            except Exception as e:
                analysis[category] = {
                    'article_count': len(data['articles']),
                    'classification_quality': {
                        'avg_confidence': avg_confidence,
                        'error': str(e)
                    }
                }

        return analysis

    def _print_category_insights(self, category_analysis):
        """
        Enhanced insights printing including classification confidence.
        """
        print("\n" + "="*60)
        print("📊 CONTENT CLASSIFICATION ANALYSIS")
        print("="*60)

        total_articles = sum(data.get('article_count', 0)
                             for data in category_analysis.values())

        for category, data in category_analysis.items():
            if 'error' in data:
                continue

            count = data['article_count']
            percentage = (count / total_articles *
                          100) if total_articles > 0 else 0

            print(
                f"\n📂 {category.upper()}: {count} articles ({percentage:.1f}%)")

            # Classification confidence metrics
            if 'classification_quality' in data:
                qual = data['classification_quality']
                avg_conf = qual.get('avg_confidence', 0)
                min_conf = qual.get('min_confidence', 0)
                max_conf = qual.get('max_confidence', 0)

                print(f"   🎯 Classification Confidence:")
                print(f"      Average: {avg_conf:.4f}")
                print(f"      Range: {min_conf:.4f} - {max_conf:.4f}")

                # Confidence quality indicator
                if avg_conf > 0.8:
                    print(f"      Quality: 🟢 High confidence")
                elif avg_conf > 0.6:
                    print(f"      Quality: 🟡 Medium confidence")
                else:
                    print(f"      Quality: 🔴 Low confidence (review needed)")

            # Show detailed classification for low-confidence items
            if 'classification_details' in data:
                low_conf_items = [item for item in data['classification_details']
                                  if item['confidence'] < 0.7]
                if low_conf_items:
                    print(f"   ⚠️  Low confidence classifications:")
                    for item in low_conf_items[:2]:  # Show top 2
                        print(
                            f"      • {item['title']} (conf: {item['confidence']:.3f})")
                        top_3 = item.get('top_3', [])
                        if len(top_3) >= 2:
                            print(
                                f"        Alternative: {top_3[1][0]} ({top_3[1][1]:.3f})")

            # Original summarization metrics
            if 'extractive_scores' in data:
                ex_rouge = data['extractive_scores'].get('rouge_rouge-1_f', 0)
                ab_rouge = data['abstractive_scores'].get('rouge_rouge-1_f', 0)
                hy_rouge = data['hybrid_scores'].get('rouge_rouge-1_f', 0)

                print(f"   📈 ROUGE-1 F1 Scores:")
                print(f"      Extractive: {ex_rouge:.4f}")
                print(f"      Abstractive: {ab_rouge:.4f}")
                print(f"      Hybrid: {hy_rouge:.4f}")

                best_method = max([
                    ('Extractive', ex_rouge),
                    ('Abstractive', ab_rouge),
                    ('Hybrid', hy_rouge)
                ], key=lambda x: x[1])
                print(
                    f"   🏆 Best method: {best_method[0]} ({best_method[1]:.4f})")

            print("="*60)
