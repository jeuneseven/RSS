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
        :param min_content_words: Minimum words to accept an article (skip if too short)
        :param timeout: Timeout (seconds) for fetching remote articles
        """
        self.min_content_words = min_content_words
        self.timeout = timeout

    def parse(self, rss_path_or_url, max_articles=5):
        """
        Parse RSS (from local file or URL). Returns a list of dicts:
        [{title:..., content:..., link:...}, ...]
        :param rss_path_or_url: Local XML path or URL
        :param max_articles: Maximum number of articles to extract
        :return: List[Dict]
        """
        # Load RSS
        if rss_path_or_url.startswith("http"):
            feed = feedparser.parse(rss_path_or_url)
        else:
            with open(rss_path_or_url, 'r', encoding='utf-8') as f:
                feed = feedparser.parse(f.read())
        results = []
        for entry in feed.entries[:max_articles]:
            title = entry.get('title', '')
            # Try multiple fields for content
            content = entry.get('content', [{'value': ''}])[0].get(
                'value', '') or entry.get('summary', '') or entry.get('description', '')
            content = self.clean_html(content)
            link = entry.get('link', '')
            # Fallback: fetch full article if content is too short
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
            # Final check for minimum content
            if len(content.split()) >= self.min_content_words:
                results.append(
                    {'title': title, 'content': content, 'link': link})
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
