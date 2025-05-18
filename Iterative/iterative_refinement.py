"""
Three-Phase Iterative Article Summarization

This script processes articles by dividing them into three parts based on sentence count:
1. First part: TextRank extractive summarization + BART abstractive fusion
2. Second part: LexRank extractive summarization + BART abstractive fusion
3. Third part: LSA extractive summarization + BART abstractive fusion

The three summaries are then concatenated and passed to BART for final refinement.
Each phase and the final result are evaluated using ROUGE and BERTScore metrics.
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


class ThreePhaseArticleSummarizer:
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

    def three_phase_summarization(self, text, num_sentences_per_part=3):
        """
        Three-phase summarization approach:
        1. Divides text into 3 parts by sentence count
        2. Applies TextRank to part 1, LexRank to part 2, LSA to part 3
        3. Creates hybrid summaries for each part with BART
        4. Combines all summaries and refines with BART
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)

        # If text is too short, return the full text
        if total_sentences <= 5:
            return text, {}, {}

        # Calculate sentences per part (at least 3 sentences per part if possible)
        part_size = max(3, total_sentences // 3)

        # Create the three parts
        part1 = " ".join(sentences[:part_size])
        part2 = " ".join(sentences[part_size:2*part_size])
        part3 = " ".join(sentences[2*part_size:])

        print("\n=== Processing Text in Three Phases ===")
        print(f"Total sentences: {total_sentences}")
        print(f"Part 1: {len(sent_tokenize(part1))} sentences")
        print(f"Part 2: {len(sent_tokenize(part2))} sentences")
        print(f"Part 3: {len(sent_tokenize(part3))} sentences")

        results = {
            'phase1': {'extractive': {}, 'abstractive': {}, 'hybrid': {}},
            'phase2': {'extractive': {}, 'abstractive': {}, 'hybrid': {}},
            'phase3': {'extractive': {}, 'abstractive': {}, 'hybrid': {}},
            'final': {}
        }

        # Phase 1: TextRank + BART for first part
        print("\n--- Phase 1: TextRank + BART ---")
        # Calculate number of sentences to extract (proportionally)
        num_sentences = max(2, len(sent_tokenize(part1)) // 3)

        # TextRank extractive summarization
        print("Applying TextRank to Part 1...")
        extractive_summary1 = self.textrank_summarization(part1, num_sentences)

        # BART abstractive summarization
        print("Applying BART to Part 1...")
        abstractive_summary1 = self.abstractive_summarization(part1)

        # Hybrid summarization
        print("Creating hybrid summary for Part 1...")
        hybrid_summary1 = self.hybrid_summarization(
            part1, part1, extractive_summary1, abstractive_summary1)

        # Evaluate summaries for Part 1
        rouge_ext1 = self._evaluate_rouge(extractive_summary1, part1)
        rouge_abs1 = self._evaluate_rouge(abstractive_summary1, part1)
        rouge_hybrid1 = self._evaluate_rouge(hybrid_summary1, part1)

        bertscore_ext1 = self._evaluate_bertscore(extractive_summary1, part1)
        bertscore_abs1 = self._evaluate_bertscore(abstractive_summary1, part1)
        bertscore_hybrid1 = self._evaluate_bertscore(hybrid_summary1, part1)

        # Store results for Phase 1
        results['phase1']['extractive']['rouge'] = rouge_ext1
        results['phase1']['extractive']['bertscore'] = bertscore_ext1
        results['phase1']['abstractive']['rouge'] = rouge_abs1
        results['phase1']['abstractive']['bertscore'] = bertscore_abs1
        results['phase1']['hybrid']['rouge'] = rouge_hybrid1
        results['phase1']['hybrid']['bertscore'] = bertscore_hybrid1

        # Phase 2: LexRank + BART for second part
        print("\n--- Phase 2: LexRank + BART ---")
        # Calculate number of sentences to extract
        num_sentences = max(2, len(sent_tokenize(part2)) // 3)

        # LexRank extractive summarization
        print("Applying LexRank to Part 2...")
        extractive_summary2 = self.lexrank_summarization(part2, num_sentences)

        # BART abstractive summarization
        print("Applying BART to Part 2...")
        abstractive_summary2 = self.abstractive_summarization(part2)

        # Hybrid summarization
        print("Creating hybrid summary for Part 2...")
        hybrid_summary2 = self.hybrid_summarization(
            part2, part2, extractive_summary2, abstractive_summary2)

        # Evaluate summaries for Part 2
        rouge_ext2 = self._evaluate_rouge(extractive_summary2, part2)
        rouge_abs2 = self._evaluate_rouge(abstractive_summary2, part2)
        rouge_hybrid2 = self._evaluate_rouge(hybrid_summary2, part2)

        bertscore_ext2 = self._evaluate_bertscore(extractive_summary2, part2)
        bertscore_abs2 = self._evaluate_bertscore(abstractive_summary2, part2)
        bertscore_hybrid2 = self._evaluate_bertscore(hybrid_summary2, part2)

        # Store results for Phase 2
        results['phase2']['extractive']['rouge'] = rouge_ext2
        results['phase2']['extractive']['bertscore'] = bertscore_ext2
        results['phase2']['abstractive']['rouge'] = rouge_abs2
        results['phase2']['abstractive']['bertscore'] = bertscore_abs2
        results['phase2']['hybrid']['rouge'] = rouge_hybrid2
        results['phase2']['hybrid']['bertscore'] = bertscore_hybrid2

        # Phase 3: LSA + BART for third part
        print("\n--- Phase 3: LSA + BART ---")
        # Calculate number of sentences to extract
        num_sentences = max(2, len(sent_tokenize(part3)) // 3)

        # LSA extractive summarization
        print("Applying LSA to Part 3...")
        extractive_summary3 = self.lsa_summarization(part3, num_sentences)

        # BART abstractive summarization
        print("Applying BART to Part 3...")
        abstractive_summary3 = self.abstractive_summarization(part3)

        # Hybrid summarization
        print("Creating hybrid summary for Part 3...")
        hybrid_summary3 = self.hybrid_summarization(
            part3, part3, extractive_summary3, abstractive_summary3)

        # Evaluate summaries for Part 3
        rouge_ext3 = self._evaluate_rouge(extractive_summary3, part3)
        rouge_abs3 = self._evaluate_rouge(abstractive_summary3, part3)
        rouge_hybrid3 = self._evaluate_rouge(hybrid_summary3, part3)

        bertscore_ext3 = self._evaluate_bertscore(extractive_summary3, part3)
        bertscore_abs3 = self._evaluate_bertscore(abstractive_summary3, part3)
        bertscore_hybrid3 = self._evaluate_bertscore(hybrid_summary3, part3)

        # Store results for Phase 3
        results['phase3']['extractive']['rouge'] = rouge_ext3
        results['phase3']['extractive']['bertscore'] = bertscore_ext3
        results['phase3']['abstractive']['rouge'] = rouge_abs3
        results['phase3']['abstractive']['bertscore'] = bertscore_abs3
        results['phase3']['hybrid']['rouge'] = rouge_hybrid3
        results['phase3']['hybrid']['bertscore'] = bertscore_hybrid3

        # Final Phase: Combine and Refine
        print("\n--- Final Phase: Combination and Refinement ---")
        # Combine the three hybrid summaries
        combined_summary = f"{hybrid_summary1} {hybrid_summary2} {hybrid_summary3}"

        # Final refinement with BART
        print("Refining combined summary with BART...")
        final_summary = self.abstractive_summarization(
            combined_summary,
            max_length=min(250, len(combined_summary.split()) // 2),
            min_length=min(100, len(combined_summary.split()) // 4)
        )

        # Evaluate final summary
        rouge_final = self._evaluate_rouge(final_summary, text)
        bertscore_final = self._evaluate_bertscore(final_summary, text)

        # Store results for final summary
        results['final']['rouge'] = rouge_final
        results['final']['bertscore'] = bertscore_final

        return final_summary, combined_summary, results

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

    def _calculate_average_scores(self, all_results):
        """
        Calculate and return average ROUGE and BERTScore metrics across all articles
        """
        # Initialize dictionary to store average scores
        avg_scores = {
            'phase1': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'phase2': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'phase3': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'final': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
        }

        # Count articles processed
        num_articles = len(all_results)

        # Sum scores across all articles
        for article_result in all_results:
            results = article_result['results']

            # Phase 1
            if 'phase1' in results:
                # Extractive
                avg_scores['phase1']['extractive']['rouge-1'] += results['phase1']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase1']['extractive']['rouge-2'] += results['phase1']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase1']['extractive']['rouge-l'] += results['phase1']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase1']['extractive']['bertscore'] += results['phase1']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase1']['abstractive']['rouge-1'] += results['phase1']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase1']['abstractive']['rouge-2'] += results['phase1']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase1']['abstractive']['rouge-l'] += results['phase1']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase1']['abstractive']['bertscore'] += results['phase1']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase1']['hybrid']['rouge-1'] += results['phase1']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase1']['hybrid']['rouge-2'] += results['phase1']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase1']['hybrid']['rouge-l'] += results['phase1']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase1']['hybrid']['bertscore'] += results['phase1']['hybrid']['bertscore']['f1']

            # Phase 2
            if 'phase2' in results:
                # Extractive
                avg_scores['phase2']['extractive']['rouge-1'] += results['phase2']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase2']['extractive']['rouge-2'] += results['phase2']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase2']['extractive']['rouge-l'] += results['phase2']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase2']['extractive']['bertscore'] += results['phase2']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase2']['abstractive']['rouge-1'] += results['phase2']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase2']['abstractive']['rouge-2'] += results['phase2']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase2']['abstractive']['rouge-l'] += results['phase2']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase2']['abstractive']['bertscore'] += results['phase2']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase2']['hybrid']['rouge-1'] += results['phase2']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase2']['hybrid']['rouge-2'] += results['phase2']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase2']['hybrid']['rouge-l'] += results['phase2']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase2']['hybrid']['bertscore'] += results['phase2']['hybrid']['bertscore']['f1']

            # Phase 3
            if 'phase3' in results:
                # Extractive
                avg_scores['phase3']['extractive']['rouge-1'] += results['phase3']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase3']['extractive']['rouge-2'] += results['phase3']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase3']['extractive']['rouge-l'] += results['phase3']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase3']['extractive']['bertscore'] += results['phase3']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase3']['abstractive']['rouge-1'] += results['phase3']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase3']['abstractive']['rouge-2'] += results['phase3']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase3']['abstractive']['rouge-l'] += results['phase3']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase3']['abstractive']['bertscore'] += results['phase3']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase3']['hybrid']['rouge-1'] += results['phase3']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase3']['hybrid']['rouge-2'] += results['phase3']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase3']['hybrid']['rouge-l'] += results['phase3']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase3']['hybrid']['bertscore'] += results['phase3']['hybrid']['bertscore']['f1']

            # Final
            if 'final' in results:
                avg_scores['final']['rouge-1'] += results['final']['rouge']['rouge-1']['f']
                avg_scores['final']['rouge-2'] += results['final']['rouge']['rouge-2']['f']
                avg_scores['final']['rouge-l'] += results['final']['rouge']['rouge-l']['f']
                avg_scores['final']['bertscore'] += results['final']['bertscore']['f1']

        # Calculate averages
        for phase in ['phase1', 'phase2', 'phase3']:
            for method in ['extractive', 'abstractive', 'hybrid']:
                for metric in avg_scores[phase][method]:
                    avg_scores[phase][method][metric] /= num_articles

        for metric in avg_scores['final']:
            avg_scores['final'][metric] /= num_articles

        # Print average results
        print("\n=== Average Scores Across All Articles ===")

        print("\n--- Phase 1: TextRank + BART ---")
        print(f"TextRank - ROUGE-1: {avg_scores['phase1']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase1']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase1']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['hybrid']['bertscore']:.4f}")

        print("\n--- Phase 2: LexRank + BART ---")
        print(f"LexRank - ROUGE-1: {avg_scores['phase2']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase2']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase2']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['hybrid']['bertscore']:.4f}")

        print("\n--- Phase 3: LSA + BART ---")
        print(f"LSA - ROUGE-1: {avg_scores['phase3']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase3']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase3']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['hybrid']['bertscore']:.4f}")

        print("\n--- Final Summary ---")
        print(f"ROUGE-1: {avg_scores['final']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['final']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['final']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['final']['bertscore']:.4f}")

        return avg_scores

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

        # Store results for visualization
        all_results = []

        print(f"Processing {len(articles)} articles from the RSS feed.")

        # Process each article
        for i, article in enumerate(articles):
            print(f"\nArticle {i+1}: {article.title}")

            # Get article content
            content = self.get_article_content(article)

            if not content or len(content.split()) < 30:
                print(f"  Insufficient content for article {i+1}, skipping.")
                continue

            # Apply three-phase summarization
            final_summary, combined_summary, results = self.three_phase_summarization(
                content)

            # Add to results for visualization
            all_results.append({
                'title': article.title,
                'results': results,
                'final_summary': final_summary,
                'original_length': len(content.split()),
                'final_length': len(final_summary.split())
            })

            # Print summaries
            print("\n=== Original Article Length ===")
            print(
                f"Words: {len(content.split())}, Sentences: {len(sent_tokenize(content))}")

            print("\n=== Final Summary ===")
            print(final_summary)
            print(
                f"Words: {len(final_summary.split())}, Sentences: {len(sent_tokenize(final_summary))}")

            print("\n=== Evaluation Results ===")
            print(
                f"ROUGE-1 F1: {results['final']['rouge']['rouge-1']['f']:.4f}")
            print(
                f"ROUGE-2 F1: {results['final']['rouge']['rouge-2']['f']:.4f}")
            print(
                f"ROUGE-L F1: {results['final']['rouge']['rouge-l']['f']:.4f}")
            print(f"BERTScore F1: {results['final']['bertscore']['f1']:.4f}")

        # Calculate average scores across all articles
        if all_results:
            avg_scores = self._calculate_average_scores(all_results)

            # Visualize results
            self._visualize_results(avg_scores, all_results)

        return all_results

    def _visualize_results(self, all_results):
        """
        Visualize the results of the three-phase summarization
        """
        # Calculate average scores across all articles
        avg_scores = {
            'phase1': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'phase2': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'phase3': {
                'extractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'abstractive': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0},
                'hybrid': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
            },
            'final': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bertscore': 0}
        }

        # Count articles processed
        num_articles = len(all_results)

        # Sum scores across all articles
        for article_result in all_results:
            results = article_result['results']

            # Phase 1
            if 'phase1' in results:
                # Extractive
                avg_scores['phase1']['extractive']['rouge-1'] += results['phase1']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase1']['extractive']['rouge-2'] += results['phase1']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase1']['extractive']['rouge-l'] += results['phase1']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase1']['extractive']['bertscore'] += results['phase1']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase1']['abstractive']['rouge-1'] += results['phase1']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase1']['abstractive']['rouge-2'] += results['phase1']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase1']['abstractive']['rouge-l'] += results['phase1']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase1']['abstractive']['bertscore'] += results['phase1']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase1']['hybrid']['rouge-1'] += results['phase1']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase1']['hybrid']['rouge-2'] += results['phase1']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase1']['hybrid']['rouge-l'] += results['phase1']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase1']['hybrid']['bertscore'] += results['phase1']['hybrid']['bertscore']['f1']

            # Phase 2
            if 'phase2' in results:
                # Extractive
                avg_scores['phase2']['extractive']['rouge-1'] += results['phase2']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase2']['extractive']['rouge-2'] += results['phase2']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase2']['extractive']['rouge-l'] += results['phase2']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase2']['extractive']['bertscore'] += results['phase2']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase2']['abstractive']['rouge-1'] += results['phase2']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase2']['abstractive']['rouge-2'] += results['phase2']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase2']['abstractive']['rouge-l'] += results['phase2']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase2']['abstractive']['bertscore'] += results['phase2']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase2']['hybrid']['rouge-1'] += results['phase2']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase2']['hybrid']['rouge-2'] += results['phase2']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase2']['hybrid']['rouge-l'] += results['phase2']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase2']['hybrid']['bertscore'] += results['phase2']['hybrid']['bertscore']['f1']

            # Phase 3
            if 'phase3' in results:
                # Extractive
                avg_scores['phase3']['extractive']['rouge-1'] += results['phase3']['extractive']['rouge']['rouge-1']['f']
                avg_scores['phase3']['extractive']['rouge-2'] += results['phase3']['extractive']['rouge']['rouge-2']['f']
                avg_scores['phase3']['extractive']['rouge-l'] += results['phase3']['extractive']['rouge']['rouge-l']['f']
                avg_scores['phase3']['extractive']['bertscore'] += results['phase3']['extractive']['bertscore']['f1']

                # Abstractive
                avg_scores['phase3']['abstractive']['rouge-1'] += results['phase3']['abstractive']['rouge']['rouge-1']['f']
                avg_scores['phase3']['abstractive']['rouge-2'] += results['phase3']['abstractive']['rouge']['rouge-2']['f']
                avg_scores['phase3']['abstractive']['rouge-l'] += results['phase3']['abstractive']['rouge']['rouge-l']['f']
                avg_scores['phase3']['abstractive']['bertscore'] += results['phase3']['abstractive']['bertscore']['f1']

                # Hybrid
                avg_scores['phase3']['hybrid']['rouge-1'] += results['phase3']['hybrid']['rouge']['rouge-1']['f']
                avg_scores['phase3']['hybrid']['rouge-2'] += results['phase3']['hybrid']['rouge']['rouge-2']['f']
                avg_scores['phase3']['hybrid']['rouge-l'] += results['phase3']['hybrid']['rouge']['rouge-l']['f']
                avg_scores['phase3']['hybrid']['bertscore'] += results['phase3']['hybrid']['bertscore']['f1']

            # Final
            if 'final' in results:
                avg_scores['final']['rouge-1'] += results['final']['rouge']['rouge-1']['f']
                avg_scores['final']['rouge-2'] += results['final']['rouge']['rouge-2']['f']
                avg_scores['final']['rouge-l'] += results['final']['rouge']['rouge-l']['f']
                avg_scores['final']['bertscore'] += results['final']['bertscore']['f1']

        # Calculate averages
        for phase in ['phase1', 'phase2', 'phase3']:
            for method in ['extractive', 'abstractive', 'hybrid']:
                for metric in avg_scores[phase][method]:
                    avg_scores[phase][method][metric] /= num_articles

        for metric in avg_scores['final']:
            avg_scores['final'][metric] /= num_articles

        # Print average results
        print("\n=== Average Scores Across All Articles ===")

        print("\n--- Phase 1: TextRank + BART ---")
        print(f"TextRank - ROUGE-1: {avg_scores['phase1']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase1']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase1']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase1']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase1']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase1']['hybrid']['bertscore']:.4f}")

        print("\n--- Phase 2: LexRank + BART ---")
        print(f"LexRank - ROUGE-1: {avg_scores['phase2']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase2']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase2']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase2']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase2']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase2']['hybrid']['bertscore']:.4f}")

        print("\n--- Phase 3: LSA + BART ---")
        print(f"LSA - ROUGE-1: {avg_scores['phase3']['extractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['extractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['extractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['extractive']['bertscore']:.4f}")
        print(f"BART - ROUGE-1: {avg_scores['phase3']['abstractive']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['abstractive']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['abstractive']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['abstractive']['bertscore']:.4f}")
        print(f"Hybrid - ROUGE-1: {avg_scores['phase3']['hybrid']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['phase3']['hybrid']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['phase3']['hybrid']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['phase3']['hybrid']['bertscore']:.4f}")

        print("\n--- Final Summary ---")
        print(f"ROUGE-1: {avg_scores['final']['rouge-1']:.4f}, "
              f"ROUGE-2: {avg_scores['final']['rouge-2']:.4f}, "
              f"ROUGE-L: {avg_scores['final']['rouge-l']:.4f}, "
              f"BERTScore: {avg_scores['final']['bertscore']:.4f}")

        # Create visualizations
        self._create_comparison_charts(avg_scores, all_results)

    def _create_comparison_charts(self, avg_scores, all_results):
        """
        Create charts comparing the performance of different summarization methods
        """
        # Create bar chart to compare ROUGE-1 scores across all phases
        plt.figure(figsize=(15, 10))

        # Data for extractive methods
        extractive_methods = [
            'TextRank (Phase 1)', 'LexRank (Phase 2)', 'LSA (Phase 3)']
        extractive_scores = [
            avg_scores['phase1']['extractive']['rouge-1'],
            avg_scores['phase2']['extractive']['rouge-1'],
            avg_scores['phase3']['extractive']['rouge-1']
        ]

        # Data for abstractive (BART) in each phase
        abstractive_scores = [
            avg_scores['phase1']['abstractive']['rouge-1'],
            avg_scores['phase2']['abstractive']['rouge-1'],
            avg_scores['phase3']['abstractive']['rouge-1']
        ]

        # Data for hybrid approaches in each phase
        hybrid_scores = [
            avg_scores['phase1']['hybrid']['rouge-1'],
            avg_scores['phase2']['hybrid']['rouge-1'],
            avg_scores['phase3']['hybrid']['rouge-1']
        ]

        # Add final summary score
        final_score = avg_scores['final']['rouge-1']

        # Bar positions
        x = np.arange(len(extractive_methods))
        width = 0.25

        # Create the bars
        plt.bar(x - width, extractive_scores, width,
                label='Extractive', color='skyblue')
        plt.bar(x, abstractive_scores, width,
                label='Abstractive (BART)', color='lightgreen')
        plt.bar(x + width, hybrid_scores, width,
                label='Hybrid', color='salmon')

        # Add a horizontal line for final summary score
        plt.axhline(y=final_score, color='purple', linestyle='--', linewidth=2)
        plt.text(x[-1] + width + 0.2, final_score, f'Final Summary: {final_score:.4f}',
                 color='purple', fontweight='bold')

        # Customize the chart
        plt.xlabel('Summarization Methods by Phase')
        plt.ylabel('ROUGE-1 F1 Score')
        plt.title('Comparison of ROUGE-1 Scores Across All Phases')
        plt.xticks(x, extractive_methods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add score labels on top of bars
        for i, v in enumerate(extractive_scores):
            plt.text(i - width, v + 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(abstractive_scores):
            plt.text(i, v + 0.01, f'{v:.4f}',
                     ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(hybrid_scores):
            plt.text(i + width, v + 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('rouge1_comparison.png', dpi=300, bbox_inches='tight')

        # Create a second chart for BERTScore comparison
        plt.figure(figsize=(15, 10))

        # Data for BERTScores
        bertscore_extractive = [
            avg_scores['phase1']['extractive']['bertscore'],
            avg_scores['phase2']['extractive']['bertscore'],
            avg_scores['phase3']['extractive']['bertscore']
        ]

        bertscore_abstractive = [
            avg_scores['phase1']['abstractive']['bertscore'],
            avg_scores['phase2']['abstractive']['bertscore'],
            avg_scores['phase3']['abstractive']['bertscore']
        ]

        bertscore_hybrid = [
            avg_scores['phase1']['hybrid']['bertscore'],
            avg_scores['phase2']['hybrid']['bertscore'],
            avg_scores['phase3']['hybrid']['bertscore']
        ]

        # Add final summary BERTScore
        final_bertscore = avg_scores['final']['bertscore']

        # Create the bars
        plt.bar(x - width, bertscore_extractive, width,
                label='Extractive', color='skyblue')
        plt.bar(x, bertscore_abstractive, width,
                label='Abstractive (BART)', color='lightgreen')
        plt.bar(x + width, bertscore_hybrid, width,
                label='Hybrid', color='salmon')

        # Add a horizontal line for final summary score
        plt.axhline(y=final_bertscore, color='purple',
                    linestyle='--', linewidth=2)
        plt.text(x[-1] + width + 0.2, final_bertscore, f'Final Summary: {final_bertscore:.4f}',
                 color='purple', fontweight='bold')

        # Customize the chart
        plt.xlabel('Summarization Methods by Phase')
        plt.ylabel('BERTScore F1')
        plt.title('Comparison of BERTScore F1 Across All Phases')
        plt.xticks(x, extractive_methods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add score labels on top of bars
        for i, v in enumerate(bertscore_extractive):
            plt.text(i - width, v + 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(bertscore_abstractive):
            plt.text(i, v + 0.01, f'{v:.4f}',
                     ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(bertscore_hybrid):
            plt.text(i + width, v + 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('bertscore_comparison.png', dpi=300, bbox_inches='tight')

        # Create a comprehensive chart showing all metrics for final summary
        plt.figure(figsize=(12, 8))

        # Metrics for final summary
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
        final_scores = [
            avg_scores['final']['rouge-1'],
            avg_scores['final']['rouge-2'],
            avg_scores['final']['rouge-l'],
            avg_scores['final']['bertscore']
        ]

        # Create bar chart for final summary metrics
        bar_positions = np.arange(len(metrics))
        plt.bar(bar_positions, final_scores, color='purple', alpha=0.7)

        # Add labels on top of bars
        for i, v in enumerate(final_scores):
            plt.text(i, v + 0.01, f'{v:.4f}',
                     ha='center', va='bottom', fontsize=10)

        # Customize chart
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics for Final Three-Phase Summary')
        plt.xticks(bar_positions, metrics)
        plt.ylim(0, max(final_scores) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('final_summary_metrics.png', dpi=300, bbox_inches='tight')

        # Additional comprehensive visualization - all phases and metrics in one chart
        plt.figure(figsize=(18, 12))

        # Categories for grouped bar chart
        categories = [
            'Phase 1\nTextRank', 'Phase 1\nBART', 'Phase 1\nHybrid',
            'Phase 2\nLexRank', 'Phase 2\nBART', 'Phase 2\nHybrid',
            'Phase 3\nLSA', 'Phase 3\nBART', 'Phase 3\nHybrid',
            'Final\nSummary'
        ]

        # ROUGE-1 scores for all methods
        rouge1_scores = [
            avg_scores['phase1']['extractive']['rouge-1'],
            avg_scores['phase1']['abstractive']['rouge-1'],
            avg_scores['phase1']['hybrid']['rouge-1'],
            avg_scores['phase2']['extractive']['rouge-1'],
            avg_scores['phase2']['abstractive']['rouge-1'],
            avg_scores['phase2']['hybrid']['rouge-1'],
            avg_scores['phase3']['extractive']['rouge-1'],
            avg_scores['phase3']['abstractive']['rouge-1'],
            avg_scores['phase3']['hybrid']['rouge-1'],
            avg_scores['final']['rouge-1']
        ]

        # BERTScore scores for all methods
        bertscore_scores = [
            avg_scores['phase1']['extractive']['bertscore'],
            avg_scores['phase1']['abstractive']['bertscore'],
            avg_scores['phase1']['hybrid']['bertscore'],
            avg_scores['phase2']['extractive']['bertscore'],
            avg_scores['phase2']['abstractive']['bertscore'],
            avg_scores['phase2']['hybrid']['bertscore'],
            avg_scores['phase3']['extractive']['bertscore'],
            avg_scores['phase3']['abstractive']['bertscore'],
            avg_scores['phase3']['hybrid']['bertscore'],
            avg_scores['final']['bertscore']
        ]

        # Bar positions
        x = np.arange(len(categories))
        width = 0.35

        # Create bars
        plt.bar(x - width/2, rouge1_scores, width,
                label='ROUGE-1 F1', color='blue', alpha=0.7)
        plt.bar(x + width/2, bertscore_scores, width,
                label='BERTScore F1', color='green', alpha=0.7)

        # Add labels and customize
        plt.xlabel('Summarization Methods by Phase')
        plt.ylabel('Score')
        plt.title('Comprehensive Comparison of Three-Phase Iterative Summarization')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Highlight the final summary
        plt.axvspan(9 - 0.5, 9 + 0.5, color='yellow', alpha=0.2)

        # Add a text box with explanation
        plt.figtext(0.5, 0.01,
                    'Phase 1: TextRank + BART on first third of article\n'
                    'Phase 2: LexRank + BART on second third of article\n'
                    'Phase 3: LSA + BART on final third of article\n'
                    'Final: Combined and refined with BART',
                    ha='center', fontsize=12, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('comprehensive_comparison.png',
                    dpi=300, bbox_inches='tight')

        print("\nVisualizations saved as:")
        print("- rouge1_comparison.png")
        print("- bertscore_comparison.png")
        print("- final_summary_metrics.png")
        print("- comprehensive_comparison.png")


def main():
    """
    Main function to run the three-phase article summarization
    """
    # Initialize the summarizer
    print("Initializing Three-Phase Article Summarizer...")
    summarizer = ThreePhaseArticleSummarizer()

    # Get RSS feed URL from user
    rss_url = input("Enter RSS feed URL or local file path: ")

    # Get number of articles to process
    try:
        num_articles = int(
            input("Enter number of articles to process (default: 3): "))
    except ValueError:
        num_articles = 3
        print("Invalid input, using default of 3 articles.")

    # Process the feed
    print(f"\nProcessing RSS feed: {rss_url}")
    start_time = time.time()
    summarizer.process_feed(rss_url, num_article=num_articles)
    end_time = time.time()

    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
