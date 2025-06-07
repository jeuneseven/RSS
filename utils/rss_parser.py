"""
utils/rss_parser.py

RSS feed parser and cleaner utility.
- Supports both local and remote RSS feeds.
- Extracts cleaned article content (HTML stripping, fallback fields, etc.).
- Provides (title, content, link) per article for downstream processing.
"""

import feedparser
import requests
import re
from bs4 import BeautifulSoup


class RSSParser:

    def __init__(self, min_content_words=30, timeout=10):
        """
        Initialize RSS parser with zero-shot classification.
        """
        self.min_content_words = min_content_words
        self.timeout = timeout

        # Initialize zero-shot classifier
        try:
            from models.pretrained_classifier import PretrainedClassifier
            print("🔄 Initializing zero-shot classifier...")
            self.classifier = PretrainedClassifier()
            self.classification_enabled = True
            print("✅ Zero-shot classifier ready")

            # Test classification to ensure it works
            test_result = self.classifier.classify_with_scores(
                "This is a test article about Samsung technology products and AI appliances.")
            if test_result['success']:
                print(
                    f"✅ Classification test passed: {test_result['predicted_label']} (conf: {test_result['confidence']:.3f})")
            else:
                print(
                    f"❌ Classification test failed: {test_result.get('error', 'Unknown error')}")
                self.classification_enabled = False

        except Exception as e:
            print(f"❌ Zero-shot classifier initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.classification_enabled = False

    def parse(self, rss_path_or_url, max_articles=5):
        """
        Parse RSS feed with zero-shot classification and detailed logging.
        """
        print(f"🔄 Parsing RSS: {rss_path_or_url}")
        print(
            f"📊 Zero-shot classification enabled: {self.classification_enabled}")

        # Load RSS
        if rss_path_or_url.startswith("http"):
            feed = feedparser.parse(rss_path_or_url)
        else:
            with open(rss_path_or_url, 'r', encoding='utf-8') as f:
                feed = feedparser.parse(f.read())

        results = []
        for i, entry in enumerate(feed.entries[:max_articles]):
            print(
                f"\n📄 Processing article {i+1}/{min(max_articles, len(feed.entries))}")

            title = entry.get('title', '')
            print(f"   Title: {title[:60]}{'...' if len(title) > 60 else ''}")

            content = entry.get('content', [{'value': ''}])[0].get(
                'value', '') or entry.get('summary', '') or entry.get('description', '')
            content = self.clean_html(content)
            link = entry.get('link', '')

            print(f"   Content length: {len(content.split())} words")

            # Fallback: fetch full article if content is too short
            if len(content.split()) < self.min_content_words and link:
                print(f"   🔄 Content too short, fetching from URL...")
                try:
                    resp = requests.get(link, timeout=self.timeout)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        tags = soup.find_all(['article', 'div', 'section'], class_=re.compile(
                            r'(article|content|post|entry)'))
                        if tags:
                            largest = max(
                                tags, key=lambda t: len(t.get_text()))
                            content = self.clean_html(largest.get_text())
                            print(
                                f"   ✅ Fetched content: {len(content.split())} words")
                except Exception as e:
                    print(f"   ❌ Failed to fetch content: {e}")

            # Final check for minimum content
            if len(content.split()) >= self.min_content_words:
                article_data = {'title': title,
                                'content': content, 'link': link}

                # Zero-shot classification with detailed logging
                if self.classification_enabled:
                    print(f"   🤖 Running zero-shot classification...")

                    # Use title + first part of content for classification
                    classification_text = f"{title}. {' '.join(content.split()[:200])}"
                    print(
                        f"   📝 Classification text: {len(classification_text.split())} words")

                    classification_result = self.classifier.classify_with_scores(
                        classification_text)

                    if classification_result['success']:
                        predicted_label = classification_result['predicted_label']
                        confidence = classification_result['confidence']

                        article_data['category'] = predicted_label
                        article_data['classification_confidence'] = confidence
                        article_data['classification_scores'] = classification_result['all_scores']
                        article_data['top_3_predictions'] = classification_result['top_3_predictions']
                        article_data['classification_method'] = classification_result['method']

                        print(
                            f"   ✅ Classification: {predicted_label} (confidence: {confidence:.4f})")
                        print(
                            f"   📊 Top 3: {classification_result['top_3_predictions'][:3]}")

                        # Show detailed scores for debugging
                        print(f"   🔍 All scores:")
                        for cat, score in sorted(classification_result['all_scores'].items(),
                                                 key=lambda x: x[1], reverse=True):
                            print(f"      {cat}: {score:.4f}")

                    else:
                        print(
                            f"   ❌ Classification failed: {classification_result.get('error', 'Unknown error')}")
                        article_data['category'] = 'unknown'
                        article_data['classification_confidence'] = 0.0
                        article_data['classification_error'] = classification_result.get(
                            'error', 'Classification failed')
                        article_data['classification_method'] = 'failed'
                else:
                    print(f"   ⚠️ Classification disabled")
                    article_data['category'] = 'unknown'
                    article_data['classification_confidence'] = 0.0
                    article_data['classification_method'] = 'disabled'

                results.append(article_data)
                print(f"   ✅ Article processed successfully")
            else:
                print(
                    f"   ❌ Content too short ({len(content.split())} words), skipping")

        print(f"\n📊 Processed {len(results)} articles successfully")
        return results

    @staticmethod
    def clean_html(html_content):
        """
        Remove HTML tags and clean text.
        :param html_content: Raw HTML string
        :return: Cleaned text
        """
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
