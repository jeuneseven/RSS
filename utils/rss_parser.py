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
        Initialize RSS parser with optional classification capability.
        Classification is enabled silently without affecting existing interface.
        """
        self.min_content_words = min_content_words
        self.timeout = timeout

        # Silent classification initialization
        try:
            from models.classification import BertClassifier
            self.classifier = BertClassifier(num_labels=6)
            self.classifier.set_label_map({
                0: "technology", 1: "sports", 2: "politics",
                3: "business", 4: "entertainment", 5: "health"
            })
            self.classification_enabled = True
            print("✓ Classification enabled")
        except Exception as e:
            print(f"⚠ Classification not available: {e}")
            self.classification_enabled = False

    def parse(self, rss_path_or_url, max_articles=5):
        """
        Parse RSS feed with silent classification enhancement.
        Original parsing logic remains unchanged.
        """
        # Original RSS parsing logic (unchanged)
        if rss_path_or_url.startswith("http"):
            feed = feedparser.parse(rss_path_or_url)
        else:
            with open(rss_path_or_url, 'r', encoding='utf-8') as f:
                feed = feedparser.parse(f.read())

        results = []
        for entry in feed.entries[:max_articles]:
            title = entry.get('title', '')
            content = entry.get('content', [{'value': ''}])[0].get(
                'value', '') or entry.get('summary', '') or entry.get('description', '')
            content = self.clean_html(content)
            link = entry.get('link', '')

            # Original content validation and fallback logic (unchanged)
            if len(content.split()) < self.min_content_words and link:
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
                except Exception:
                    pass

            # Final content validation (unchanged)
            if len(content.split()) >= self.min_content_words:
                article_data = {'title': title,
                                'content': content, 'link': link}

                # Silent classification enhancement - doesn't break existing code
                if self.classification_enabled:
                    try:
                        category = self.classifier.classify(content)
                        article_data['category'] = category
                    except Exception:
                        article_data['category'] = 'unknown'
                else:
                    article_data['category'] = 'unknown'

                results.append(article_data)

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
