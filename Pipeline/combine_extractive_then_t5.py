"""
RSS Feed Summarization and Evaluation - 5 Methods Comparison with T5

This script analyzes articles from an RSS feed and generates five types of summaries:
1. TextRank - extractive summarization using TextRank algorithm
2. LexRank - extractive summarization using LexRank algorithm
3. LSA - extractive summarization using LSA algorithm
4. T5 - abstractive summarization using T5 model
5. Hybrid - combines multiple approaches

Each summary type is evaluated using ROUGE and BERTScore metrics against the original text.
"""

import feedparser
import requests
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BertTokenizer,
    BertModel
)
from rouge import Rouge
from bert_score import score as bert_score
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import time

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class RSSFeedSummarizer:
    def __init__(self):
        # Initialize models
        self.stop_words = set(stopwords.words('english'))

        # Initialize transformer models
        print("Loading models...")
        # T5 model for abstractive summarization
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

        # BERT model for BERTScore
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Initialize ROUGE evaluator
        self.rouge = Rouge()
        print("Models loaded successfully.")

    def fetch_rss(self, rss_url):
        """
        Fetch and parse RSS feed from a URL or local file
        """
        try:
            if rss_url.startswith(('http://', 'https://')):
                feed = feedparser.parse(rss_url)
            else:
                # Handle local file
                with open(rss_url, 'r') as f:
                    feed = feedparser.parse(f.read())

            return feed
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return None

    def clean_html(self, html_content):
        """
        Remove HTML tags and clean text
        """
        if not html_content:
            return ""

        # Use BeautifulSoup to remove HTML tags
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=' ')

        # Clean the text
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def get_article_content(self, article):
        """
        Extract content from an article in the RSS feed
        """
        # Try to get content from different fields that might contain the article text
        content = ""
        if hasattr(article, 'content') and article.get('content'):
            content = article.content[0].value
        elif article.get('summary'):
            content = article.summary
        elif article.get('description'):
            content = article.description

        # Clean HTML content
        content = self.clean_html(content)

        # If content is too short, try to fetch the full article
        if len(content.split()) < 100 and article.get('link'):
            try:
                response = requests.get(article.link, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Try to find the main content (this is a simple approach, might need refinement)
                    article_tags = soup.find_all(['article', 'div', 'section'], class_=re.compile(
                        r'(article|content|post|entry)'))
                    if article_tags:
                        # Use the largest tag assuming it's the main content
                        largest_tag = max(
                            article_tags, key=lambda x: len(x.get_text()))
                        content = self.clean_html(largest_tag.get_text())
            except Exception as e:
                print(f"Error fetching full article: {e}")

        return content

    def textrank_summary(self, text, num_sentences=5):
        """
        Generate an extractive summary using TextRank algorithm
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)

        # If there are fewer sentences than requested, return all sentences
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j])

        # Generate graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Sort sentences by score and select top n
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Extract original indices for top sentences to maintain original order
        top_indices = [sentences.index(ranked_sentences[i][1]) for i in range(
            min(num_sentences, len(ranked_sentences)))]
        top_indices.sort()

        # Create summary by joining selected sentences in original order
        summary = " ".join([sentences[i] for i in top_indices])

        return summary

    def lexrank_summary(self, text, num_sentences=5):
        """
        Generate an extractive summary using LexRank algorithm
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j])

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ranked = sorted(((scores[i], s)
                        for i, s in enumerate(sentences)), reverse=True)

        # Maintain original order
        top_indices = [sentences.index(ranked[i][1]) for i in range(
            min(num_sentences, len(ranked)))]
        top_indices.sort()

        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def lsa_summary(self, text, num_sentences=5):
        """
        Generate an extractive summary using LSA algorithm
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        svd = TruncatedSVD(n_components=1)
        svd.fit(X)
        scores = svd.components_[0]
        ranked = sorted(((scores[i], s)
                        for i, s in enumerate(sentences)), reverse=True)

        # Maintain original order
        top_indices = [sentences.index(ranked[i][1]) for i in range(
            min(num_sentences, len(ranked)))]
        top_indices.sort()

        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def _sentence_similarity(self, sent1, sent2):
        """
        Compute similarity between two sentences using cosine similarity
        """
        # Tokenize and filter words
        words1 = [word.lower() for word in nltk.word_tokenize(sent1)
                  if word.lower() not in self.stop_words]
        words2 = [word.lower() for word in nltk.word_tokenize(sent2)
                  if word.lower() not in self.stop_words]

        # Create a set of all words
        all_words = list(set(words1 + words2))

        # Create word vectors
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]

        # Calculate cosine similarity
        if not any(vector1) or not any(vector2):
            return 0.0

        return 1 - cosine_distance(vector1, vector2)

    def t5_summary(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using T5 model directly on original text
        """
        input_text = "summarize: " + text
        inputs = self.t5_tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = self.t5_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.t5_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)

        return summary

    def hybrid_summary(self, text, textrank_summary, lexrank_summary, lsa_summary, t5_summary):
        """
        Generate a hybrid summary by combining the best elements from all methods
        """
        # Evaluate each method's performance
        methods = {
            'textrank': textrank_summary,
            'lexrank': lexrank_summary,
            'lsa': lsa_summary,
            't5': t5_summary
        }

        method_scores = {}
        for method, summary in methods.items():
            rouge_score = self._evaluate_rouge(summary, text)
            bert_score = self._evaluate_bertscore(summary, text)
            # Combined score
            method_scores[method] = (
                rouge_score['rouge-1']['f'] + bert_score['f1']) / 2

        # Find the best performing method
        best_method = max(method_scores, key=method_scores.get)
        best_summary = methods[best_method]

        # Get sentences from all summaries
        all_sentences = []
        for summary in methods.values():
            all_sentences.extend(sent_tokenize(summary))

        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sent in all_sentences:
            normalized = sent.strip().lower()
            # Filter very short sentences
            if normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_sentences.append(sent)

        # If we have the best method summary and some unique content, combine them
        if len(unique_sentences) > len(sent_tokenize(best_summary)):
            # Start with best method summary
            hybrid_sentences = sent_tokenize(best_summary)

            # Add unique valuable sentences from other methods
            for sent in unique_sentences:
                if sent not in best_summary:
                    # Check if this sentence adds value
                    is_valuable = True
                    for existing_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, existing_sent) > 0.7:
                            is_valuable = False
                            break

                    if is_valuable and len(hybrid_sentences) < 6:  # Limit total length
                        hybrid_sentences.append(sent)

            hybrid_summary = " ".join(hybrid_sentences)
        else:
            hybrid_summary = best_summary

        return hybrid_summary

    def _evaluate_rouge(self, summary, reference):
        """
        Evaluate summary using ROUGE metrics
        """
        if not summary or not reference:
            # Return zero scores if either input is empty
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

        try:
            # Calculate ROUGE scores
            scores = self.rouge.get_scores(summary, reference)[0]
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            # Return zero scores in case of error
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

    def _evaluate_bertscore(self, summary, reference):
        """
        Evaluate summary using BERTScore metrics
        """
        if not summary or not reference:
            # Return zero scores if either input is empty
            return {'p': 0.0, 'r': 0.0, 'f1': 0.0}

        try:
            # Calculate BERTScore
            P, R, F1 = bert_score([summary], [reference], lang='en')
            return {
                'p': P.item(),
                'r': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            # Return zero scores in case of error
            return {'p': 0.0, 'r': 0.0, 'f1': 0.0}

    def process_feed(self, rss_url, num_articles=3):
        """
        Process the RSS feed and generate summaries using all 5 methods
        """
        # Fetch and parse the RSS feed
        feed = self.fetch_rss(rss_url)

        if not feed or not feed.entries:
            print("No articles found in the RSS feed.")
            return

        # Limit to the first n articles
        articles = feed.entries[:min(num_articles, len(feed.entries))]

        # Store results for each method
        results = {
            'textrank': {'rouge': {}, 'bertscore': {}},
            'lexrank': {'rouge': {}, 'bertscore': {}},
            'lsa': {'rouge': {}, 'bertscore': {}},
            't5': {'rouge': {}, 'bertscore': {}},
            'hybrid': {'rouge': {}, 'bertscore': {}}
        }

        print(f"Processing {len(articles)} articles from the RSS feed.")

        # Process each article
        for i, article in enumerate(articles):
            print(f"\nArticle {i+1}: {article.title}")

            # Get article content
            content = self.get_article_content(article)

            if not content or len(content.split()) < 30:
                print(f"  Insufficient content for article {i+1}, skipping.")
                continue

            # Generate summaries using all 5 methods
            print("  Generating TextRank summary...")
            textrank_summary = self.textrank_summary(content)

            print("  Generating LexRank summary...")
            lexrank_summary = self.lexrank_summary(content)

            print("  Generating LSA summary...")
            lsa_summary = self.lsa_summary(content)

            print("  Generating T5 summary...")
            t5_summary = self.t5_summary(content)

            print("  Generating Hybrid summary...")
            hybrid_summary = self.hybrid_summary(
                content, textrank_summary, lexrank_summary, lsa_summary, t5_summary
            )

            # Evaluate all summaries against original text
            print("  Evaluating summaries...")

            summaries = {
                'textrank': textrank_summary,
                'lexrank': lexrank_summary,
                'lsa': lsa_summary,
                't5': t5_summary,
                'hybrid': hybrid_summary
            }

            for method, summary in summaries.items():
                # ROUGE evaluation
                rouge_score = self._evaluate_rouge(summary, content)
                results[method]['rouge'][i] = rouge_score

                # BERTScore evaluation
                bertscore_result = self._evaluate_bertscore(summary, content)
                results[method]['bertscore'][i] = bertscore_result

            # Print results for this article
            print(f"\n  === Summary Results for Article {i+1} ===")
            print(f"  Original length: {len(content.split())} words")

            for method, summary in summaries.items():
                print(
                    f"  {method.capitalize()} summary: {len(summary.split())} words")

            print("\n  --- ROUGE-1 F1 Scores ---")
            for method in summaries.keys():
                score = results[method]['rouge'][i]['rouge-1']['f']
                print(f"  {method.capitalize()}: {score:.4f}")

            print("\n  --- BERTScore F1 ---")
            for method in summaries.keys():
                score = results[method]['bertscore'][i]['f1']
                print(f"  {method.capitalize()}: {score:.4f}")

            # Print the summaries
            for method, summary in summaries.items():
                print(f"\n  --- {method.capitalize()} Summary ---")
                print(f"  {summary}")

        # Calculate and display average scores
        self._calculate_average_scores(results, len(articles))

        # Visualize results
        self._visualize_results(results)

        return results

    def _calculate_average_scores(self, results, num_articles):
        """
        Calculate and display average scores across all articles for all 5 methods
        """
        methods = ['textrank', 'lexrank', 'lsa', 't5', 'hybrid']
        avg_scores = {}

        for method in methods:
            avg_scores[method] = {
                'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0
            }

        # Sum scores for each metric
        for method in methods:
            for i in range(num_articles):
                if i in results[method]['rouge']:
                    avg_scores[method]['rouge-1'] += results[method]['rouge'][i]['rouge-1']['f']
                    avg_scores[method]['rouge-2'] += results[method]['rouge'][i]['rouge-2']['f']
                    avg_scores[method]['rouge-l'] += results[method]['rouge'][i]['rouge-l']['f']

                if i in results[method]['bertscore']:
                    avg_scores[method]['bertscore'] += results[method]['bertscore'][i]['f1']

        # Calculate averages
        for method in avg_scores:
            for metric in avg_scores[method]:
                avg_scores[method][metric] /= num_articles

        # Display average results
        print("\n=== Average Scores Across All Articles (5 Methods Comparison) ===")

        print("\n--- ROUGE Scores ---")
        for method in methods:
            print(
                f"{method.capitalize()} - ROUGE-1: {avg_scores[method]['rouge-1']:.4f}, "
                f"ROUGE-2: {avg_scores[method]['rouge-2']:.4f}, "
                f"ROUGE-L: {avg_scores[method]['rouge-l']:.4f}")

        print("\n--- BERTScore ---")
        for method in methods:
            print(
                f"{method.capitalize()} - F1: {avg_scores[method]['bertscore']:.4f}")

        return avg_scores

    def _visualize_results(self, results):
        """
        Create visualizations comparing the performance of all 5 summarization methods
        """
        # Method names with clear labels
        methods = ['TextRank', 'LexRank', 'LSA', 'T5', 'Hybrid']
        method_keys = ['textrank', 'lexrank', 'lsa', 't5', 'hybrid']

        # Calculate average scores for each method
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bertscore_scores = []

        for method_key in method_keys:
            # ROUGE-1 scores
            scores = [results[method_key]['rouge'][i]['rouge-1']['f']
                      for i in results[method_key]['rouge']]
            rouge1_scores.append(np.mean(scores))

            # ROUGE-2 scores
            scores = [results[method_key]['rouge'][i]['rouge-2']['f']
                      for i in results[method_key]['rouge']]
            rouge2_scores.append(np.mean(scores))

            # ROUGE-L scores
            scores = [results[method_key]['rouge'][i]['rouge-l']['f']
                      for i in results[method_key]['rouge']]
            rougeL_scores.append(np.mean(scores))

            # BERTScore scores
            scores = [results[method_key]['bertscore'][i]['f1']
                      for i in results[method_key]['bertscore']]
            bertscore_scores.append(np.mean(scores))

        # Create a DataFrame for better visualization
        data = {
            'Method': methods,
            'ROUGE-1': rouge1_scores,
            'ROUGE-2': rouge2_scores,
            'ROUGE-L': rougeL_scores,
            'BERTScore': bertscore_scores
        }
        df = pd.DataFrame(data)

        # Print the comparison table
        print("\n=== Performance Comparison of All 5 Summarization Methods ===")
        print(df.to_string(index=False))

        # Create enhanced bar charts
        plt.figure(figsize=(16, 12))

        # Set colors for each method
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        # ROUGE-1 scores
        plt.subplot(2, 2, 1)
        bars = plt.bar(methods, rouge1_scores, color=colors, width=0.6)
        plt.title('ROUGE-1 F1 Scores Comparison',
                  fontsize=14, fontweight='bold')
        plt.ylim(0, max(rouge1_scores) * 1.2)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45)

        # ROUGE-2 scores
        plt.subplot(2, 2, 2)
        bars = plt.bar(methods, rouge2_scores, color=colors, width=0.6)
        plt.title('ROUGE-2 F1 Scores Comparison',
                  fontsize=14, fontweight='bold')
        plt.ylim(0, max(rouge2_scores) * 1.2)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45)

        # ROUGE-L scores
        plt.subplot(2, 2, 3)
        bars = plt.bar(methods, rougeL_scores, color=colors, width=0.6)
        plt.title('ROUGE-L F1 Scores Comparison',
                  fontsize=14, fontweight='bold')
        plt.ylim(0, max(rougeL_scores) * 1.2)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45)

        # BERTScore scores
        plt.subplot(2, 2, 4)
        bars = plt.bar(methods, bertscore_scores, color=colors, width=0.6)
        plt.title('BERTScore F1 Scores Comparison',
                  fontsize=14, fontweight='bold')
        plt.ylim(0, max(bertscore_scores) * 1.2)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45)

        # Add comprehensive title and method explanations
        plt.suptitle('Comparison of 5 Summarization Methods - All Evaluated Against Original Text',
                     fontsize=18, fontweight='bold', y=0.98)

        plt.figtext(0.5, 0.02,
                    'TextRank: Graph-based extractive (PageRank algorithm)\n'
                    'LexRank: Graph-based extractive (sentence similarity ranking)\n'
                    'LSA: Latent Semantic Analysis extractive (dimensionality reduction)\n'
                    'T5: Text-to-Text Transfer Transformer abstractive (pre-trained neural model)\n'
                    'Hybrid: Adaptive combination selecting best performing elements from all methods\n'
                    'All ROUGE and BERTScore evaluations are against the original article text',
                    ha='center', fontsize=11,
                    bbox={"facecolor": "lightgrey", "alpha": 0.7, "pad": 8})

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig('combine_extractive_then_t5.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("\nComprehensive 5-method comparison with T5 visualization saved as 'combine_extractive_then_t5.png'")


def main():
    """
    Main function to run the RSS feed summarization and evaluation for 5 methods with T5
    """
    # Initialize the summarizer
    print("Initializing RSS Feed Summarizer for 5-method comparison with T5...")
    summarizer = RSSFeedSummarizer()

    # Get RSS feed URL from user
    rss_url = input("Enter RSS feed URL or local file path: ")

    # Process the feed
    print(f"\nProcessing RSS feed: {rss_url}")
    print("Will compare: TextRank, LexRank, LSA, T5, and Hybrid methods")
    start_time = time.time()
    summarizer.process_feed(rss_url, num_articles=5)
    end_time = time.time()

    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
