"""
RSS Feed Summarization and Evaluation

This script analyzes articles from an RSS feed and generates three types of summaries:
1. Extractive summaries via TextRank
2. Abstractive summaries via BART
3. Hybrid summaries by using TextRank output as prompt for BART

Each summary type is evaluated using ROUGE metrics, and a comparison bar chart of average ROUGE-1 F1 scores is generated and saved.
"""

import feedparser
import requests
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge import Rouge
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from bert_score import score as bert_score

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class RSSFeedSummarizer:
    def __init__(self):
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))

        # Load BART model and tokenizer
        print("Loading BART models...")
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')

        # ROUGE evaluator
        self.rouge = Rouge()
        print("Models loaded successfully.")

    def fetch_rss(self, rss_url):
        """Fetch and parse RSS feed from a URL or local file"""
        try:
            if rss_url.startswith(('http://', 'https://')):
                return feedparser.parse(rss_url)
            with open(rss_url, 'r') as f:
                return feedparser.parse(f.read())
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return None

    def clean_html(self, html_content):
        """Remove HTML tags and clean text"""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=' ')
        return re.sub(r'\s+', ' ', text).strip()

    def get_article_content(self, article):
        """Extract and clean article content"""
        content = ''
        if article.get('content'):
            content = article.content[0].value
        elif article.get('summary'):
            content = article.summary
        elif article.get('description'):
            content = article.description
        content = self.clean_html(content)
        if len(content.split()) < 100 and article.get('link'):
            try:
                resp = requests.get(article.link, timeout=10)
                if resp.ok:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    tags = soup.find_all(['article', 'div', 'section'], class_=re.compile(
                        r'(article|content|post|entry)'))
                    if tags:
                        largest = max(tags, key=lambda t: len(t.get_text()))
                        content = self.clean_html(largest.get_text())
            except Exception:
                pass
        return content

    def textrank_summarization(self, text, num_sentences=5):
        """Extractive summary via TextRank"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        # Build similarity matrix
        tfidf = TfidfVectorizer(stop_words='english')
        m = tfidf.fit_transform(sentences)
        sim_matrix = (m * m.T).toarray()
        np.fill_diagonal(sim_matrix, 0)
        # Rank sentences
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[
            :num_sentences]
        return ' '.join([sentences[i] for i in sorted(ranked)])

    def bart_summarization(self, text, max_length=150, min_length=50):
        """Abstractive summary via BART"""
        inputs = self.bart_tokenizer(
            text, truncation=True, max_length=1024, return_tensors='pt')
        ids = self.bart_model.generate(
            inputs['input_ids'], max_length=max_length,
            min_length=min_length, length_penalty=2.0,
            num_beams=4, early_stopping=True
        )
        return self.bart_tokenizer.decode(ids[0], skip_special_tokens=True)

    def hybrid_summarization(self, text):
        """Hybrid summary: use TextRank output as prompt for BART"""
        ext = self.textrank_summarization(text)
        return self.bart_summarization(ext)

    def _evaluate_rouge(self, summary, reference):
        """Compute ROUGE scores"""
        if not summary or not reference:
            return {'rouge-1': {'f': 0}}
        return self.rouge.get_scores(summary, reference)[0]

    def _evaluate_bertscore(self, summary, reference):
        """Compute BERTScore F1"""
        if not summary or not reference:
            return 0.0
        P, R, F1 = bert_score([summary], [reference],
                              lang='en', rescale_with_baseline=True)
        return F1[0].item()

    def process_feed(self, rss_url, num_articles=3):
        """Fetch feed, generate summaries, evaluate, and plot comparison"""
        feed = self.fetch_rss(rss_url)
        if not feed or not feed.entries:
            print("No entries found in feed.")
            return

        tr_scores, bart_scores, hyb_scores = [], [], []
        bert_tr_scores, bert_bart_scores, bert_hyb_scores = [], [], []
        for entry in feed.entries[:num_articles]:
            text = self.get_article_content(entry)
            tr = self.textrank_summarization(text)
            bart = self.bart_summarization(text)
            hyb = self.hybrid_summarization(text)

            ref = text
            tr_scores.append(self._evaluate_rouge(tr, ref)['rouge-1']['f'])
            bart_scores.append(self._evaluate_rouge(bart, ref)['rouge-1']['f'])
            hyb_scores.append(self._evaluate_rouge(hyb, ref)['rouge-1']['f'])
            bert_tr_scores.append(self._evaluate_bertscore(tr, ref))
            bert_bart_scores.append(self._evaluate_bertscore(bart, ref))
            bert_hyb_scores.append(self._evaluate_bertscore(hyb, ref))

        # Calculate average scores
        methods = ['TextRank', 'BART', 'Hybrid']
        avg_rouge = [np.mean(tr_scores), np.mean(
            bart_scores), np.mean(hyb_scores)]
        avg_bert = [np.mean(bert_tr_scores), np.mean(
            bert_bart_scores), np.mean(bert_hyb_scores)]

        # Plot
        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, avg_rouge, width, label='ROUGE-1 F1')
        bars2 = ax.bar(x + width/2, avg_bert, width, label='BERTScore F1')

        # Annotate bars
        for i in range(len(methods)):
            ax.text(x[i] - width/2, avg_rouge[i] + 0.01,
                    f'{avg_rouge[i]:.4f}', ha='center')
            ax.text(x[i] + width/2, avg_bert[i] + 0.01,
                    f'{avg_bert[i]:.4f}', ha='center')

        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('F1 Score')
        ax.set_title('Average ROUGE-1 and BERTScore F1 Scores by Method')
        ax.legend()
        plt.tight_layout()
        plt.savefig('textrank_bart_prompt_hybrid.png', dpi=300)
        plt.close()
        print("Comparison chart saved as 'textrank_bart_prompt_hybrid.png'")


def main():
    summarizer = RSSFeedSummarizer()
    url = input("Enter RSS feed URL or file path: ")
    summarizer.process_feed(url, num_articles=5)


if __name__ == '__main__':
    main()
