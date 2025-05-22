"""
RSS Feed Summarization and Evaluation

This script analyzes articles from an RSS feed and generates summaries using:
1. Extractive methods (selects best from 3 algorithms):
   - TextRank: Graph-based ranking algorithm
   - LexRank: Graph-based algorithm using TF-IDF cosine similarity
   - LSA: Latent Semantic Analysis for topic extraction
2. Abstractive methods (selects best from 3 models):
   - BART: Facebook's BART model for conditional generation
   - T5: Google's Text-to-Text Transfer Transformer
   - Pegasus: Google's pre-trained model for abstractive summarization
3. Hybrid method: combines the best extractive and best abstractive approaches

Each method is evaluated using ROUGE and BERTScore metrics to determine the best performers.
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
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from scipy.sparse.linalg import svds

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
        print("  Loading BART model...")
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')

        # T5 model for abstractive summarization
        print("  Loading T5 model...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Pegasus model for abstractive summarization
        print("  Loading Pegasus model...")
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(
            'google/pegasus-xsum')
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
            'google/pegasus-xsum')

        # BERT model for BERTScore
        print("  Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Initialize ROUGE evaluator
        self.rouge = Rouge()
        print("All models loaded successfully.")

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

        # Clean the text - replace multiple spaces with a single space
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
                    # Try to find the main content
                    article_tags = soup.find_all(['article', 'div', 'section'],
                                                 class_=re.compile(r'(article|content|post|entry)'))
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

    def bart_summarization(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using BART model
        """
        try:
            # Truncate text if it's too long for BART
            input_ids = self.bart_tokenizer.encode(
                text, truncation=True, max_length=1024, return_tensors="pt")

            # Check if input is too short
            if input_ids.shape[1] < 10:
                return "Summary not available - input too short."

            # Generate summary
            summary_ids = self.bart_model.generate(
                input_ids,
                max_length=max_length,
                min_length=min(min_length, input_ids.shape[1]),
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.bart_tokenizer.pad_token_id,
                eos_token_id=self.bart_tokenizer.eos_token_id
            )

            summary = self.bart_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary if summary.strip() else "Summary generation failed."

        except Exception as e:
            print(f"Error in BART summarization: {e}")
            return "BART summarization failed due to model constraints."

    def t5_summarization(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using T5 model
        """
        try:
            # T5 requires a task prefix
            input_text = f"summarize: {text}"

            # Truncate text if it's too long for T5
            input_ids = self.t5_tokenizer.encode(
                input_text, truncation=True, max_length=512, return_tensors="pt")

            # Check if input is too short
            if input_ids.shape[1] < 10:
                return "Summary not available - input too short."

            # Generate summary
            summary_ids = self.t5_model.generate(
                input_ids,
                max_length=max_length,
                min_length=min(min_length, input_ids.shape[1]),
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.t5_tokenizer.pad_token_id,
                eos_token_id=self.t5_tokenizer.eos_token_id
            )

            summary = self.t5_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary if summary.strip() else "Summary generation failed."

        except Exception as e:
            print(f"Error in T5 summarization: {e}")
            return "T5 summarization failed due to model constraints."

    def pegasus_summarization(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using Pegasus model
        """
        try:
            # Pegasus has a smaller max position embeddings limit, so use 512 instead of 1024
            input_ids = self.pegasus_tokenizer.encode(
                text, truncation=True, max_length=512, return_tensors="pt")

            # Check if input is too short
            if input_ids.shape[1] < 10:
                return "Summary not available - input too short."

            # Generate summary with error handling
            summary_ids = self.pegasus_model.generate(
                input_ids,
                max_length=max_length,
                # Ensure min_length doesn't exceed input length
                min_length=min(min_length, input_ids.shape[1]),
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.pegasus_tokenizer.pad_token_id,
                eos_token_id=self.pegasus_tokenizer.eos_token_id
            )

            summary = self.pegasus_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary if summary.strip() else "Summary generation failed."

        except Exception as e:
            print(f"Error in Pegasus summarization: {e}")
            return "Pegasus summarization failed due to model constraints."

    def hybrid_summarization(self, text, original_text, best_extractive_summary, best_abstractive_summary):
        """
        Generate a hybrid summary that combines the best extractive and best abstractive approaches
        to achieve higher ROUGE and BERTScore values
        """
        # Get ROUGE scores for both summaries
        rouge_ext = self._evaluate_rouge(
            best_extractive_summary, original_text)
        rouge_abs = self._evaluate_rouge(
            best_abstractive_summary, original_text)

        # Get BERTScore for both summaries
        bertscore_ext = self._evaluate_bertscore(
            best_extractive_summary, original_text)
        bertscore_abs = self._evaluate_bertscore(
            best_abstractive_summary, original_text)

        # Calculate weights based on performance
        ext_weight = (rouge_ext['rouge-1']['f'] + bertscore_ext['f1']) / 2
        abs_weight = (rouge_abs['rouge-1']['f'] + bertscore_abs['f1']) / 2

        # Normalize weights
        total = ext_weight + abs_weight
        if total > 0:
            ext_weight = ext_weight / total
            abs_weight = abs_weight / total
        else:
            ext_weight = abs_weight = 0.5

        # Combine the summaries based on their strengths
        if ext_weight > abs_weight * 1.2:  # If extractive is significantly better
            # Use extractive as base and enhance with abstractive elements
            sentences_ext = sent_tokenize(best_extractive_summary)
            sentences_abs = sent_tokenize(best_abstractive_summary)

            # Find unique information in abstractive summary
            unique_abs_sentences = []
            for abs_sent in sentences_abs:
                is_unique = True
                for ext_sent in sentences_ext:
                    if self._sentence_similarity(abs_sent, ext_sent) > 0.6:
                        is_unique = False
                        break
                if is_unique:
                    unique_abs_sentences.append(abs_sent)

            # Combine extractive summary with unique abstractive sentences
            hybrid_summary = best_extractive_summary
            if unique_abs_sentences:
                hybrid_summary += " " + " ".join(unique_abs_sentences[:2])

        elif abs_weight > ext_weight * 1.2:  # If abstractive is significantly better
            # Use abstractive as base and enhance with key extractive sentences
            sentences_ext = sent_tokenize(best_extractive_summary)
            # Take top 2 sentences from extractive
            key_sentences = sentences_ext[:2]

            # Use abstractive summary as base
            hybrid_summary = best_abstractive_summary

            # Add key extractive sentences that contain unique information
            for sent in key_sentences:
                if sent not in best_abstractive_summary:
                    hybrid_summary += " " + sent
        else:
            # Both are comparable, create a more balanced hybrid
            sentences_ext = sent_tokenize(best_extractive_summary)
            sentences_abs = sent_tokenize(best_abstractive_summary)

            hybrid_sentences = []

            # Add first sentence from abstractive (often a good overview)
            if sentences_abs:
                hybrid_sentences.append(sentences_abs[0])

            # Add important details from extractive
            for i, sent in enumerate(sentences_ext):
                if i < 3:  # Limit to 3 sentences from extractive
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

            # Add remaining unique info from abstractive
            for i, sent in enumerate(sentences_abs[1:]):
                if i < 2:  # Limit to 2 more sentences from abstractive
                    is_unique = True
                    for hybrid_sent in hybrid_sentences:
                        if self._sentence_similarity(sent, hybrid_sent) > 0.7:
                            is_unique = False
                            break
                    if is_unique:
                        hybrid_sentences.append(sent)

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
                return best_extractive_summary
            else:
                return best_abstractive_summary

        return hybrid_summary

    def _evaluate_rouge(self, summary, reference):
        """
        Evaluate summary using ROUGE metrics
        """
        if not summary or not reference or summary.strip() == "" or reference.strip() == "":
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

        # Handle error messages from summarization
        error_messages = ["failed", "not available", "error", "constraints"]
        if any(msg in summary.lower() for msg in error_messages):
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

        try:
            scores = self.rouge.get_scores(summary, reference)[0]
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

    def _evaluate_bertscore(self, summary, reference):
        """
        Evaluate summary using BERTScore metrics
        """
        if not summary or not reference or summary.strip() == "" or reference.strip() == "":
            return {'p': 0.0, 'r': 0.0, 'f1': 0.0}

        # Handle error messages from summarization
        error_messages = ["failed", "not available", "error", "constraints"]
        if any(msg in summary.lower() for msg in error_messages):
            return {'p': 0.0, 'r': 0.0, 'f1': 0.0}

        try:
            P, R, F1 = bert_score([summary], [reference], lang='en')
            return {
                'p': P.item(),
                'r': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
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
            # Extractive methods
            'textrank': {'rouge': {}, 'bertscore': {}},
            'lexrank': {'rouge': {}, 'bertscore': {}},
            'lsa': {'rouge': {}, 'bertscore': {}},
            'best_extractive': {'rouge': {}, 'bertscore': {}, 'method': {}},

            # Abstractive methods
            'bart': {'rouge': {}, 'bertscore': {}},
            't5': {'rouge': {}, 'bertscore': {}},
            'pegasus': {'rouge': {}, 'bertscore': {}},
            'best_abstractive': {'rouge': {}, 'bertscore': {}, 'method': {}},

            # Hybrid method
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

            # ====================== EXTRACTIVE METHODS ======================
            print("  Generating extractive summaries...")

            print("    Generating TextRank summary...")
            textrank_summary = self.textrank_summarization(content)

            print("    Generating LexRank summary...")
            lexrank_summary = self.lexrank_summarization(content)

            print("    Generating LSA summary...")
            lsa_summary = self.lsa_summarization(content)

            # Evaluate extractive summaries
            print("  Evaluating extractive summaries...")
            rouge_textrank = self._evaluate_rouge(textrank_summary, content)
            rouge_lexrank = self._evaluate_rouge(lexrank_summary, content)
            rouge_lsa = self._evaluate_rouge(lsa_summary, content)

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
            lexrank_score = (
                rouge_lexrank['rouge-1']['f'] + bertscore_lexrank['f1']) / 2
            lsa_score = (rouge_lsa['rouge-1']['f'] + bertscore_lsa['f1']) / 2

            best_ext_score = max(textrank_score, lexrank_score, lsa_score)
            if best_ext_score == textrank_score:
                best_extractive_summary = textrank_summary
                best_ext_method = "TextRank"
                results['best_extractive']['rouge'][i] = rouge_textrank
                results['best_extractive']['bertscore'][i] = bertscore_textrank
            elif best_ext_score == lexrank_score:
                best_extractive_summary = lexrank_summary
                best_ext_method = "LexRank"
                results['best_extractive']['rouge'][i] = rouge_lexrank
                results['best_extractive']['bertscore'][i] = bertscore_lexrank
            else:
                best_extractive_summary = lsa_summary
                best_ext_method = "LSA"
                results['best_extractive']['rouge'][i] = rouge_lsa
                results['best_extractive']['bertscore'][i] = bertscore_lsa

            results['best_extractive']['method'][i] = best_ext_method

            # ====================== ABSTRACTIVE METHODS ======================
            print("  Generating abstractive summaries...")

            print("    Generating BART summary...")
            bart_summary = self.bart_summarization(content)

            print("    Generating T5 summary...")
            t5_summary = self.t5_summarization(content)

            print("    Generating Pegasus summary...")
            pegasus_summary = self.pegasus_summarization(content)

            # Evaluate abstractive summaries
            print("  Evaluating abstractive summaries...")
            rouge_bart = self._evaluate_rouge(bart_summary, content)
            rouge_t5 = self._evaluate_rouge(t5_summary, content)
            rouge_pegasus = self._evaluate_rouge(pegasus_summary, content)

            bertscore_bart = self._evaluate_bertscore(bart_summary, content)
            bertscore_t5 = self._evaluate_bertscore(t5_summary, content)
            bertscore_pegasus = self._evaluate_bertscore(
                pegasus_summary, content)

            # Store results for abstractive methods
            results['bart']['rouge'][i] = rouge_bart
            results['t5']['rouge'][i] = rouge_t5
            results['pegasus']['rouge'][i] = rouge_pegasus

            results['bart']['bertscore'][i] = bertscore_bart
            results['t5']['bertscore'][i] = bertscore_t5
            results['pegasus']['bertscore'][i] = bertscore_pegasus

            # Determine the best abstractive method for this article
            bart_score = (rouge_bart['rouge-1']
                          ['f'] + bertscore_bart['f1']) / 2
            t5_score = (rouge_t5['rouge-1']['f'] + bertscore_t5['f1']) / 2
            pegasus_score = (
                rouge_pegasus['rouge-1']['f'] + bertscore_pegasus['f1']) / 2

            best_abs_score = max(bart_score, t5_score, pegasus_score)
            if best_abs_score == bart_score:
                best_abstractive_summary = bart_summary
                best_abs_method = "BART"
                results['best_abstractive']['rouge'][i] = rouge_bart
                results['best_abstractive']['bertscore'][i] = bertscore_bart
            elif best_abs_score == t5_score:
                best_abstractive_summary = t5_summary
                best_abs_method = "T5"
                results['best_abstractive']['rouge'][i] = rouge_t5
                results['best_abstractive']['bertscore'][i] = bertscore_t5
            else:
                best_abstractive_summary = pegasus_summary
                best_abs_method = "Pegasus"
                results['best_abstractive']['rouge'][i] = rouge_pegasus
                results['best_abstractive']['bertscore'][i] = bertscore_pegasus

            results['best_abstractive']['method'][i] = best_abs_method

            # ====================== HYBRID METHOD ======================
            print(
                f"  Generating hybrid summary (using {best_ext_method} + {best_abs_method})...")
            hybrid_summary = self.hybrid_summarization(
                content, content, best_extractive_summary, best_abstractive_summary
            )

            # Evaluate hybrid summary
            rouge_hybrid = self._evaluate_rouge(hybrid_summary, content)
            bertscore_hybrid = self._evaluate_bertscore(
                hybrid_summary, content)

            results['hybrid']['rouge'][i] = rouge_hybrid
            results['hybrid']['bertscore'][i] = bertscore_hybrid

            # Print detailed results for this article
            print(f"\n  === Summary Results for Article {i+1} ===")
            print(f"  Original length: {len(content.split())} words")
            print(f"  TextRank summary: {len(textrank_summary.split())} words")
            print(f"  LexRank summary: {len(lexrank_summary.split())} words")
            print(f"  LSA summary: {len(lsa_summary.split())} words")
            print(f"  BART summary: {len(bart_summary.split())} words")
            print(f"  T5 summary: {len(t5_summary.split())} words")
            print(f"  Pegasus summary: {len(pegasus_summary.split())} words")
            print(f"  Hybrid summary: {len(hybrid_summary.split())} words")

            print("\n  --- ROUGE-1 F1 Scores ---")
            print(f"  TextRank: {rouge_textrank['rouge-1']['f']:.4f}")
            print(f"  LexRank: {rouge_lexrank['rouge-1']['f']:.4f}")
            print(f"  LSA: {rouge_lsa['rouge-1']['f']:.4f}")
            print(f"  BART: {rouge_bart['rouge-1']['f']:.4f}")
            print(f"  T5: {rouge_t5['rouge-1']['f']:.4f}")
            print(f"  Pegasus: {rouge_pegasus['rouge-1']['f']:.4f}")
            print(f"  Hybrid: {rouge_hybrid['rouge-1']['f']:.4f}")

            print("\n  --- BERTScore F1 ---")
            print(f"  TextRank: {bertscore_textrank['f1']:.4f}")
            print(f"  LexRank: {bertscore_lexrank['f1']:.4f}")
            print(f"  LSA: {bertscore_lsa['f1']:.4f}")
            print(f"  BART: {bertscore_bart['f1']:.4f}")
            print(f"  T5: {bertscore_t5['f1']:.4f}")
            print(f"  Pegasus: {bertscore_pegasus['f1']:.4f}")
            print(f"  Hybrid: {bertscore_hybrid['f1']:.4f}")

            print(
                f"\n  Best extractive method for this article: {best_ext_method}")
            print(
                f"  Best abstractive method for this article: {best_abs_method}")

            # Print the summaries
            print("\n  --- TextRank Summary ---")
            print(f"  {textrank_summary}")

            print("\n  --- LexRank Summary ---")
            print(f"  {lexrank_summary}")

            print("\n  --- LSA Summary ---")
            print(f"  {lsa_summary}")

            print("\n  --- BART Summary ---")
            print(f"  {bart_summary}")

            print("\n  --- T5 Summary ---")
            print(f"  {t5_summary}")

            print("\n  --- Pegasus Summary ---")
            print(f"  {pegasus_summary}")

            print("\n  --- Hybrid Summary ---")
            print(f"  {hybrid_summary}")

        # Calculate and display average scores
        self._calculate_average_scores(results, len(articles))

        # Visualize final results - only 3 methods
        self._visualize_final_results(results)

        return results

    def _calculate_average_scores(self, results, num_articles):
        """
        Calculate and display average scores across all articles
        """
        # All methods for comprehensive analysis
        all_methods = ['textrank', 'lexrank', 'lsa', 'best_extractive',
                       'bart', 't5', 'pegasus', 'best_abstractive', 'hybrid']

        avg_scores = {}
        for method in all_methods:
            avg_scores[method] = {'rouge-1': 0,
                                  'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}

        # Count usage of each method
        extractive_method_counts = {'TextRank': 0, 'LexRank': 0, 'LSA': 0}
        abstractive_method_counts = {'BART': 0, 'T5': 0, 'Pegasus': 0}

        for i in range(num_articles):
            if i in results['best_extractive']['method']:
                method = results['best_extractive']['method'][i]
                extractive_method_counts[method] += 1
            if i in results['best_abstractive']['method']:
                method = results['best_abstractive']['method'][i]
                abstractive_method_counts[method] += 1

        # Sum scores for each metric
        for method in all_methods:
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
                if num_articles > 0:
                    avg_scores[method][metric] /= num_articles

        # Display average results
        print("\n=== Average Scores Across All Articles ===")

        print("\n--- Extractive Methods ROUGE Scores ---")
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive']:
            method_name = method.replace('_', ' ').title()
            print(f"{method_name} - ROUGE-1: {avg_scores[method]['rouge-1']:.4f}, "
                  f"ROUGE-2: {avg_scores[method]['rouge-2']:.4f}, "
                  f"ROUGE-L: {avg_scores[method]['rouge-l']:.4f}")

        print("\n--- Abstractive Methods ROUGE Scores ---")
        for method in ['bart', 't5', 'pegasus', 'best_abstractive']:
            method_name = method.upper() if method in [
                'bart', 't5'] else method.replace('_', ' ').title()
            print(f"{method_name} - ROUGE-1: {avg_scores[method]['rouge-1']:.4f}, "
                  f"ROUGE-2: {avg_scores[method]['rouge-2']:.4f}, "
                  f"ROUGE-L: {avg_scores[method]['rouge-l']:.4f}")

        print("\n--- Hybrid Method ROUGE Scores ---")
        print(f"Hybrid - ROUGE-1: {avg_scores['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['hybrid']['rouge-l']:.4f}")

        print("\n--- Extractive Methods BERTScore ---")
        for method in ['textrank', 'lexrank', 'lsa', 'best_extractive']:
            method_name = method.replace('_', ' ').title()
            print(f"{method_name} - F1: {avg_scores[method]['bertscore']:.4f}")

        print("\n--- Abstractive Methods BERTScore ---")
        for method in ['bart', 't5', 'pegasus', 'best_abstractive']:
            method_name = method.upper() if method in [
                'bart', 't5'] else method.replace('_', ' ').title()
            print(f"{method_name} - F1: {avg_scores[method]['bertscore']:.4f}")

        print("\n--- Hybrid Method BERTScore ---")
        print(f"Hybrid - F1: {avg_scores['hybrid']['bertscore']:.4f}")

        # Display method selection statistics
        print("\n--- Best Extractive Method Usage ---")
        for method, count in extractive_method_counts.items():
            percentage = (count / num_articles) * \
                100 if num_articles > 0 else 0
            print(f"{method}: {count} articles ({percentage:.1f}%)")

        print("\n--- Best Abstractive Method Usage ---")
        for method, count in abstractive_method_counts.items():
            percentage = (count / num_articles) * \
                100 if num_articles > 0 else 0
            print(f"{method}: {count} articles ({percentage:.1f}%)")

        return avg_scores

    def _visualize_final_results(self, results):
        """
        Create visualizations comparing only the 3 final methods:
        Best Extractive, Best Abstractive, and Hybrid
        """
        # Only show 3 final methods
        methods = [
            'Best Extractive\n(Dynamic Selection)',
            'Best Abstractive\n(Dynamic Selection)',
            'Hybrid\n(Best Ext + Best Abs)'
        ]

        # Calculate average ROUGE-1 F1 scores for final methods only
        rouge1_scores = []
        for method in ['best_extractive', 'best_abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-1']['f']
                      for i in results[method]['rouge']]
            rouge1_scores.append(np.mean(scores) if scores else 0)

        # Calculate average ROUGE-2 F1 scores for final methods only
        rouge2_scores = []
        for method in ['best_extractive', 'best_abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-2']['f']
                      for i in results[method]['rouge']]
            rouge2_scores.append(np.mean(scores) if scores else 0)

        # Calculate average ROUGE-L F1 scores for final methods only
        rougeL_scores = []
        for method in ['best_extractive', 'best_abstractive', 'hybrid']:
            scores = [results[method]['rouge'][i]['rouge-l']['f']
                      for i in results[method]['rouge']]
            rougeL_scores.append(np.mean(scores) if scores else 0)

        # Calculate average BERTScore F1 scores for final methods only
        bertscore_scores = []
        for method in ['best_extractive', 'best_abstractive', 'hybrid']:
            scores = [results[method]['bertscore'][i]['f1']
                      for i in results[method]['bertscore']]
            bertscore_scores.append(np.mean(scores) if scores else 0)

        # Gather data on which methods were chosen as best
        extractive_method_counts = {'TextRank': 0, 'LexRank': 0, 'LSA': 0}
        abstractive_method_counts = {'BART': 0, 'T5': 0, 'Pegasus': 0}

        for i in results['best_extractive']['method']:
            extractive_method_counts[results['best_extractive']
                                     ['method'][i]] += 1
        for i in results['best_abstractive']['method']:
            abstractive_method_counts[results['best_abstractive']
                                      ['method'][i]] += 1

        # Create a DataFrame for better visualization with final methods only
        data = {
            'Method': methods,
            'ROUGE-1': rouge1_scores,
            'ROUGE-2': rouge2_scores,
            'ROUGE-L': rougeL_scores,
            'BERTScore': bertscore_scores
        }
        df = pd.DataFrame(data)

        # Print the table with final method comparison
        print("\n=== Final Method Performance Comparison ===")
        print(df.to_string(index=False))

        # Create bar charts with only 3 final methods
        plt.figure(figsize=(15, 10))

        # Set colors for the 3 final methods
        colors = ['#3498db', '#e74c3c', '#1abc9c']  # Blue, Red, Teal

        # ROUGE-1 scores
        plt.subplot(2, 2, 1)
        bars = plt.bar(methods, rouge1_scores, color=colors, width=0.6)
        plt.title('ROUGE-1 F1 Scores - Final Methods Comparison',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rouge1_scores) * 1.2 if max(rouge1_scores) > 0 else 1)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=0)

        # ROUGE-2 scores
        plt.subplot(2, 2, 2)
        bars = plt.bar(methods, rouge2_scores, color=colors, width=0.6)
        plt.title('ROUGE-2 F1 Scores - Final Methods Comparison',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rouge2_scores) * 1.2 if max(rouge2_scores) > 0 else 1)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=0)

        # ROUGE-L scores
        plt.subplot(2, 2, 3)
        bars = plt.bar(methods, rougeL_scores, color=colors, width=0.6)
        plt.title('ROUGE-L F1 Scores - Final Methods Comparison',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(rougeL_scores) * 1.2 if max(rougeL_scores) > 0 else 1)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=0)

        # BERTScore scores
        plt.subplot(2, 2, 4)
        bars = plt.bar(methods, bertscore_scores, color=colors, width=0.6)
        plt.title('BERTScore F1 Scores - Final Methods Comparison',
                  fontsize=12, fontweight='bold')
        plt.ylim(0, max(bertscore_scores) *
                 1.2 if max(bertscore_scores) > 0 else 1)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.4f}', ha='center', fontsize=10)
        plt.ylabel('F1 Score')
        plt.xticks(rotation=0)

        # Create pie charts showing method selection distribution
        plt.figure(figsize=(15, 6))

        # Extractive methods pie chart
        plt.subplot(1, 2, 1)
        ext_methods = list(extractive_method_counts.keys())
        ext_values = list(extractive_method_counts.values())
        if sum(ext_values) > 0:
            plt.pie(ext_values, labels=ext_methods, autopct='%1.1f%%',
                    colors=['#3498db', '#2ecc71', '#e74c3c'], startangle=90)
        plt.title('Best Extractive Method Selection Distribution',
                  fontsize=12, fontweight='bold')
        plt.axis('equal')

        # Abstractive methods pie chart
        plt.subplot(1, 2, 2)
        abs_methods = list(abstractive_method_counts.keys())
        abs_values = list(abstractive_method_counts.values())
        if sum(abs_values) > 0:
            plt.pie(abs_values, labels=abs_methods, autopct='%1.1f%%',
                    colors=['#9b59b6', '#f39c12', '#1abc9c'], startangle=90)
        plt.title('Best Abstractive Method Selection Distribution',
                  fontsize=12, fontweight='bold')
        plt.axis('equal')

        # Add a super title with explanation to the main comparison chart
        plt.figure(1)
        plt.suptitle('Final Summarization Methods Performance Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.figtext(0.5, 0.02,
                    'Best Extractive: Dynamically selects the best performing extractive algorithm (TextRank/LexRank/LSA) for each article\n'
                    'Best Abstractive: Dynamically selects the best performing abstractive model (BART/T5/Pegasus) for each article\n'
                    'Hybrid: Intelligently combines the best extractive method with the best abstractive method based on performance scores',
                    ha='center', fontsize=10, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 10})

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig('best_extractive_and_best_abstractive.png',
                    dpi=300, bbox_inches='tight')

        plt.close('all')

        print("\nVisualization saved as 'best_extractive_and_best_abstractive.png'")
        print("Charts show comparison of the 3 final methods:")
        print("1. Best Extractive (dynamically selected from TextRank/LexRank/LSA)")
        print("2. Best Abstractive (dynamically selected from BART/T5/Pegasus)")
        print("3. Hybrid (Best Extractive + Best Abstractive)")


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
