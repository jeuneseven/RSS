"""
RSS Feed Summarization and Evaluation

This script analyzes articles from an RSS feed and generates four types of summaries:
1. Extractive summaries using multiple algorithms:
   - TextRank: Graph-based ranking algorithm
   - LexRank: Graph-based algorithm using TF-IDF cosine similarity
   - LSA: Latent Semantic Analysis for topic extraction
2. Abstractive summaries - generates new text using BART
3. Hybrid summaries - combines the best extractive algorithm with abstractive approach

Each summary type is evaluated using ROUGE and BERTScore metrics, and the results are compared.
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
    BartForConditionalGeneration,
    BartTokenizer,
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from scipy.sparse.linalg import svds
from scipy import sparse

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
        # BART model for abstractive summarization
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')

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

    def textrank_summarization(self, text, num_sentences=5):
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

    def lexrank_summarization(self, text, num_sentences=5):
        """
        Generate an extractive summary using LexRank algorithm
        LexRank is a graph-based algorithm that uses TF-IDF cosine similarity
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)

        # If there are fewer sentences than requested, return all sentences
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Compute similarity matrix using TF-IDF vectors
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

        # Generate graph and apply LexRank (PageRank on the similarity matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        # Alpha is the damping factor
        scores = nx.pagerank(nx_graph, alpha=0.9)

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

    def lsa_summarization(self, text, num_sentences=5, num_topics=None):
        """
        Generate an extractive summary using Latent Semantic Analysis (LSA)
        LSA uses SVD to identify important topics and related sentences
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)

        # If there are fewer sentences than requested, return all sentences
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # If num_topics not specified, use half the number of sentences or 5, whichever is smaller
        if num_topics is None:
            num_topics = min(5, len(sentences) // 2)
            # Ensure at least 2 topics if possible
            num_topics = max(2, num_topics)

        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Apply SVD to identify topics
        u, sigma, vt = svds(tfidf_matrix, k=num_topics)

        # Get sentence scores for each topic
        sentence_scores = np.square(u).sum(axis=1)

        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()  # Sort indices to maintain original order

        # Create summary by joining selected sentences in original order
        summary = " ".join([sentences[i] for i in top_indices])

        return summary

    def abstractive_summarization(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using BART model
        """
        # Truncate text if it's too long for BART
        input_ids = self.bart_tokenizer.encode(
            text, truncation=True, max_length=1024, return_tensors="pt")

        # Generate summary
        summary_ids = self.bart_model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.bart_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)

        return summary

    def hybrid_summarization(self, text, original_text, extractive_summary, abstractive_summary):
        """
        Generate a hybrid summary that combines extractive and abstractive approaches
        to achieve higher ROUGE and BERTScore values
        """
        # Get ROUGE scores for both summaries
        rouge_ext = self._evaluate_rouge(extractive_summary, original_text)
        rouge_abs = self._evaluate_rouge(abstractive_summary, original_text)

        # Get BERTScore for both summaries
        bertscore_ext = self._evaluate_bertscore(
            extractive_summary, original_text)
        bertscore_abs = self._evaluate_bertscore(
            abstractive_summary, original_text)

        # Calculate weights based on performance
        ext_weight = (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2
        abs_weight = (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2

        # Normalize weights
        total = ext_weight + abs_weight
        ext_weight = ext_weight / total
        abs_weight = abs_weight / total

        # Combine the summaries based on their strengths
        if ext_weight > abs_weight * 1.2:  # If extractive is significantly better
            # Use extractive as base and enhance with abstractive elements
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)

            # Find unique information in abstractive summary
            unique_abs_sentences = []
            for abs_sent in sentences_abs:
                is_unique = True
                for ext_sent in sentences_ext:
                    # Similarity threshold
                    if self._sentence_similarity(abs_sent, ext_sent) > 0.6:
                        is_unique = False
                        break
                if is_unique:
                    unique_abs_sentences.append(abs_sent)

            # Combine extractive summary with unique abstractive sentences
            hybrid_summary = extractive_summary
            if unique_abs_sentences:
                # Add up to 2 unique sentences
                hybrid_summary += " " + " ".join(unique_abs_sentences[:2])

        elif abs_weight > ext_weight * 1.2:  # If abstractive is significantly better
            # Use abstractive as base and enhance with key extractive sentences
            sentences_ext = sent_tokenize(extractive_summary)

            # Find key information in extractive that might be missing in abstractive
            # Take top 2 sentences from extractive
            key_sentences = sentences_ext[:2]

            # Use abstractive summary as base
            hybrid_summary = abstractive_summary

            # Add key extractive sentences that contain unique information
            for sent in key_sentences:
                if sent not in abstractive_summary:
                    hybrid_summary += " " + sent
        else:
            # Both are comparable, create a more balanced hybrid
            # Extract key sentences from both summaries
            sentences_ext = sent_tokenize(extractive_summary)
            sentences_abs = sent_tokenize(abstractive_summary)

            # Select sentences from both summaries
            hybrid_sentences = []

            # Add first sentence from abstractive (often a good overview)
            if sentences_abs:
                hybrid_sentences.append(sentences_abs[0])

            # Add important details from extractive
            for i, sent in enumerate(sentences_ext):
                if i < 3:  # Limit to 3 sentences from extractive
                    # Check if it's not too similar to what we already have
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

            # Add remaining unique info from abstractive
            # Skip first sentence which we already added
            for i, sent in enumerate(sentences_abs[1:]):
                if i < 2:  # Limit to 2 more sentences from abstractive
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

            # Join the selected sentences
            hybrid_summary = " ".join(hybrid_sentences)

        # Final check - ensure hybrid summary is not worse than both original summaries
        rouge_hybrid = self._evaluate_rouge(hybrid_summary, original_text)
        bertscore_hybrid = self._evaluate_bertscore(
            hybrid_summary, original_text)

        hybrid_score = (rouge_hybrid['rouge-1']
                        ['f'] + bertscore_hybrid['f1']) / 2
        best_original_score = max((rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2,
                                  (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2)

        # If hybrid is worse than both originals, use the better of the two originals
        if hybrid_score < best_original_score:
            if (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) > (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']):
                return extractive_summary
            else:
                return abstractive_summary

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
        Process the RSS feed and generate summaries for the first n articles
        """
        # Fetch and parse the RSS feed
        feed = self.fetch_rss(rss_url)

        if not feed or not feed.entries:
            print("No articles found in the RSS feed.")
            return

        # Limit to the first n articles
        articles = feed.entries[:min(num_articles, len(feed.entries))]

        # Store results for each article and each method
        results = {
            'textrank': {'rouge': {}, 'bertscore': {}},
            'lexrank': {'rouge': {}, 'bertscore': {}},
            'lsa': {'rouge': {}, 'bertscore': {}},
            'best_extractive': {'rouge': {}, 'bertscore': {}, 'method': {}},
            'abstractive': {'rouge': {}, 'bertscore': {}},
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

            # Generate extractive summaries with different algorithms
            print("  Generating TextRank summary...")
            textrank_summary = self.textrank_summarization(content)

            print("  Generating LexRank summary...")
            lexrank_summary = self.lexrank_summarization(content)

            print("  Generating LSA summary...")
            lsa_summary = self.lsa_summarization(content)

            # Evaluate extractive summaries
            print("  Evaluating extractive summaries...")
            # ROUGE evaluation
            rouge_textrank = self._evaluate_rouge(textrank_summary, content)
            rouge_lexrank = self._evaluate_rouge(lexrank_summary, content)
            rouge_lsa = self._evaluate_rouge(lsa_summary, content)

            # BERTScore evaluation
            bertscore_textrank = self._evaluate_bertscore(
                textrank_summary, content)
            bertscore_lexrank = self._evaluate_bertscore(
                lexrank_summary, content)
            bertscore_lsa = self._evaluate_bertscore(lsa_summary, content)

            # Store results for extractive methods
            results['textrank']['rouge'][i] = rouge_textrank
            results['lexrank']['rouge'][i] = rouge_lexrank
            results['lsa']['rouge'][i] = rouge_lsa

            results['textrank']['bertscore'][i] = bertscore_textrank
            results['lexrank']['bertscore'][i] = bertscore_lexrank
            results['lsa']['bertscore'][i] = bertscore_lsa

            # Determine the best extractive method for this article
            textrank_score = (
                rouge_textrank['rouge-1']['f'] + bertscore_textrank['f1']) / 2
            lexrank_score = (rouge_lexrank['rouge-1']
                             ['f'] + bertscore_lexrank['f1']) / 2
            lsa_score = (rouge_lsa['rouge-1']['f'] + bertscore_lsa['f1']) / 2

            best_score = max(textrank_score, lexrank_score, lsa_score)
            if best_score == textrank_score:
                best_extractive_summary = textrank_summary
                best_method = "TextRank"
                results['best_extractive']['rouge'][i] = rouge_textrank
                results['best_extractive']['bertscore'][i] = bertscore_textrank
            elif best_score == lexrank_score:
                best_extractive_summary = lexrank_summary
                best_method = "LexRank"
                results['best_extractive']['rouge'][i] = rouge_lexrank
                results['best_extractive']['bertscore'][i] = bertscore_lexrank
            else:
                best_extractive_summary = lsa_summary
                best_method = "LSA"
                results['best_extractive']['rouge'][i] = rouge_lsa
                results['best_extractive']['bertscore'][i] = bertscore_lsa

            results['best_extractive']['method'][i] = best_method

            # Generate abstractive summary
            print("  Generating abstractive summary...")
            abstractive_summary = self.abstractive_summarization(content)

            # Evaluate abstractive summary
            rouge_abstractive = self._evaluate_rouge(
                abstractive_summary, content)
            bertscore_abstractive = self._evaluate_bertscore(
                abstractive_summary, content)

            results['abstractive']['rouge'][i] = rouge_abstractive
            results['abstractive']['bertscore'][i] = bertscore_abstractive

            # Generate hybrid summary using the best extractive method and abstractive
            print(
                f"  Generating hybrid summary (using {best_method} + BART)...")
            hybrid_summary = self.hybrid_summarization(
                content, content, best_extractive_summary, abstractive_summary
            )

            # Evaluate hybrid summary
            rouge_hybrid = self._evaluate_rouge(hybrid_summary, content)
            bertscore_hybrid = self._evaluate_bertscore(
                hybrid_summary, content)

            results['hybrid']['rouge'][i] = rouge_hybrid
            results['hybrid']['bertscore'][i] = bertscore_hybrid

            # Print results for this article
            print(f"\n  === Summary Results for Article {i+1} ===")
            print(f"  Original length: {len(content.split())} words")
            print(f"  TextRank summary: {len(textrank_summary.split())} words")
            print(f"  LexRank summary: {len(lexrank_summary.split())} words")
            print(f"  LSA summary: {len(lsa_summary.split())} words")
            print(
                f"  Abstractive summary: {len(abstractive_summary.split())} words")
            print(f"  Hybrid summary: {len(hybrid_summary.split())} words")

            print("\n  --- ROUGE-1 Scores ---")
            print(f"  TextRank: {rouge_textrank['rouge-1']['f']:.4f}")
            print(f"  LexRank: {rouge_lexrank['rouge-1']['f']:.4f}")
            print(f"  LSA: {rouge_lsa['rouge-1']['f']:.4f}")
            print(f"  Abstractive: {rouge_abstractive['rouge-1']['f']:.4f}")
            print(f"  Hybrid: {rouge_hybrid['rouge-1']['f']:.4f}")

            print("\n  --- BERTScore F1 ---")
            print(f"  TextRank: {bertscore_textrank['f1']:.4f}")
            print(f"  LexRank: {bertscore_lexrank['f1']:.4f}")
            print(f"  LSA: {bertscore_lsa['f1']:.4f}")
            print(f"  Abstractive: {bertscore_abstractive['f1']:.4f}")
            print(f"  Hybrid: {bertscore_hybrid['f1']:.4f}")

            print(
                f"\n  Best extractive method for this article: {best_method}")

            # Print the summaries
            print("\n  --- TextRank Summary ---")
            print(f"  {textrank_summary}")

            print("\n  --- LexRank Summary ---")
            print(f"  {lexrank_summary}")

            print("\n  --- LSA Summary ---")
            print(f"  {lsa_summary}")

            print("\n  --- Abstractive Summary ---")
            print(f"  {abstractive_summary}")

            print("\n  --- Hybrid Summary ---")
            print(f"  {hybrid_summary}")

        # Calculate and display average scores
        self._calculate_average_scores(results, len(articles))

        # Visualize results
        self._visualize_results(results)

        return results

    def _calculate_average_scores(self, results, num_articles):
        """
        Calculate and display average scores across all articles
        """
        avg_scores = {
            'textrank': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
            'lexrank': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
            'lsa': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
            'best_extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
            'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
            'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
        }

        # Count usage of each extractive method
        method_counts = {'TextRank': 0, 'LexRank': 0, 'LSA': 0}
        for i in range(num_articles):
            if i in results['best_extractive']['method']:
                method = results['best_extractive']['method'][i]
                method_counts[method] += 1

        # Sum scores for each metric
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
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
        print("\n=== Average Scores Across All Articles ===")

        print("\n--- ROUGE Scores ---")
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            method_name = method.replace('_', ' ').title()
            print(
                f"{method_name} - ROUGE-1: {avg_scores[method]['rouge-1']:.4f}, ROUGE-2: {avg_scores[method]['rouge-2']:.4f}, ROUGE-L: {avg_scores[method]['rouge-l']:.4f}")

        print("\n--- BERTScore ---")
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            method_name = method.replace('_', ' ').title()
            print(f"{method_name} - F1: {avg_scores[method]['bertscore']:.4f}")

        # Display best extractive method usage
        print("\n--- Best Extractive Method Usage ---")
        for method, count in method_counts.items():
            percentage = (count / num_articles) * 100
            print(f"{method}: {count} articles ({percentage:.1f}%)")

        return avg_scores

    def _visualize_results(self, results):
        """
        Create visualizations comparing the performance of different summarization methods
        with clear labels of the algorithms/models used
        """
        # Prepare data for visualization with detailed method labels
        methods = [
            'TextRank',
            'LexRank',
            'LSA',
            'Best Extractive',
            'Abstractive\n(BART)',
            'Hybrid\n(Best+BART)'
        ]

        # Calculate average ROUGE-1 F1 scores
        rouge1_scores = []
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-1']['f']
                      for i in results[method]['rouge']]
            rouge1_scores.append(np.mean(scores))

        # Calculate average ROUGE-2 F1 scores
        rouge2_scores = []
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-2']['f']
                      for i in results[method]['rouge']]
            rouge2_scores.append(np.mean(scores))

        # Calculate average ROUGE-L F1 scores
        rougeL_scores = []
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-l']['f']
                      for i in results[method]['rouge']]
            rougeL_scores.append(np.mean(scores))

        # Calculate average BERTScore F1 scores
        bertscore_scores = []
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive', 'abstractive', 'hybrid']:
            scores = [results[method]['bertscore'][i]['f1']
                      for i in results[method]['bertscore']]
            bertscore_scores.append(np.mean(scores))

        # Gather data on which extractive method was chosen as best
        best_method_counts = {'TextRank': 0, 'LexRank': 0, 'LSA': 0}
        for i in results['best_extractive']['method']:
            best_method_counts[results['best_extractive']['method'][i]] += 1

        # Create a DataFrame for better visualization with algorithm/model info
        data = {
            'Method': methods,
            'ROUGE-1': rouge1_scores,
            'ROUGE-2': rouge2_scores,
            'ROUGE-L': rougeL_scores,
            'BERTScore': bertscore_scores
        }
        df = pd.DataFrame(data)

        # Print the table with algorithm information
        print("\n=== Performance Comparison of Summarization Methods ===")
        print(df.to_string(index=False))

        # Create bar charts with algorithm/model information
        plt.figure(figsize=(16, 12))

        # Set colors for each method
        colors = ['#3498db', '#2ecc71', '#e74c3c',
                  '#f39c12', '#9b59b6', '#1abc9c']

        # ROUGE-1 scores
        plt.subplot(2, 2, 1)
        bars = plt.bar(methods, rouge1_scores, color=colors, width=0.6)
        plt.title('ROUGE-1 F1 Scores by Method',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rouge1_scores) * 1.2)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', fontsize=9)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)

        # ROUGE-2 scores
        plt.subplot(2, 2, 2)
        bars = plt.bar(methods, rouge2_scores, color=colors, width=0.6)
        plt.title('ROUGE-2 F1 Scores by Method',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rouge2_scores) * 1.2)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', fontsize=9)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)

        # ROUGE-L scores
        plt.subplot(2, 2, 3)
        bars = plt.bar(methods, rougeL_scores, color=colors, width=0.6)
        plt.title('ROUGE-L F1 Scores by Method',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rougeL_scores) * 1.2)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', fontsize=9)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)

        # BERTScore scores
        plt.subplot(2, 2, 4)
        bars = plt.bar(methods, bertscore_scores, color=colors, width=0.6)
        plt.title('BERTScore F1 Scores by Method',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(bertscore_scores) * 1.2)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', fontsize=9)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)

        # Create a pie chart showing which extractive method was chosen as best most often
        plt.figure(figsize=(10, 8))
        methods_chosen = list(best_method_counts.keys())
        values = list(best_method_counts.values())
        plt.pie(values, labels=methods_chosen, autopct='%1.1f%%',
                colors=['#3498db', '#2ecc71', '#e74c3c'], startangle=90)
        plt.title('Best Extractive Method Selection Distribution',
                  fontsize=14, fontweight='bold')
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')

        # Add a super title with explanation to the main comparison chart
        plt.figure(1)
        plt.suptitle('Comparison of Summarization Methods and Their Performance',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.figtext(0.5, 0.01,
                    'TextRank: Graph-based ranking algorithm\n'
                    'LexRank: Graph-based using TF-IDF cosine similarity\n'
                    'LSA: Latent Semantic Analysis for topic extraction\n'
                    'BART: Pre-trained transformer-based abstractive model\n'
                    'Hybrid: Combines best extractive method with BART',
                    ha='center', fontsize=11, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('best_extractive_and_bart.png',
                    dpi=300, bbox_inches='tight')

        plt.close('all')

        print("\nVisualization saved as 'best_extractive_and_bart.png")


def main():
    """
    Main function to run the RSS feed summarization and evaluation
    """
    # Initialize the summarizer
    print("Initializing RSS Feed Summarizer...")
    summarizer = RSSFeedSummarizer()

    # Get RSS feed URL from user
    rss_url = input("Enter RSS feed URL or local file path: ")

    # Get number of articles to process
    try:
        num_articles = int(
            input("Enter number of articles to process (default: 5): "))
    except ValueError:
        num_articles = 5
        print("Invalid input, using default of 5 articles.")

    # Process the feed
    print(f"\nProcessing RSS feed: {rss_url}")
    start_time = time.time()
    summarizer.process_feed(rss_url, num_articles=num_articles)
    end_time = time.time()

    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
