import feedparser
import re
import json
import traceback
from typing import Dict, List, Tuple, Any, Optional
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# For TextRank summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# For BART summarization - use lighter model for CPU inference
from transformers import pipeline

# For ROUGE evaluation
from rouge import Rouge

# Import common utilities
from utils import clean_html_text, get_entry_content


def textrank_summarization(text: str, num_sentences: int = 3) -> str:
    """
    Generate an extractive summary using TextRank algorithm

    Args:
        text: Input text to summarize
        num_sentences: Number of sentences to extract

    Returns:
        Extractive summary
    """
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    try:
        # Clean text
        clean_text = clean_html_text(text)

        # Parse the text
        parser = PlaintextParser.from_string(clean_text, Tokenizer("english"))

        # Use TextRank algorithm
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count=num_sentences)

        # Join the sentences into a string
        summary_text = " ".join(str(sentence) for sentence in summary)

        return summary_text

    except Exception as e:
        print(f"Error in TextRank summarization: {e}")
        # Fallback to first few sentences
        sentences = text.split('. ')
        return '. '.join(sentences[:num_sentences]) + ('.' if not sentences[0].endswith('.') else '')


def bart_summarization(text: str, max_length: int = 100, min_length: int = 30) -> str:
    """
    Generate an abstractive summary using BART model with improved parameters

    Args:
        text: Input text to summarize
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary

    Returns:
        Abstractive summary
    """
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    try:
        # Clean text
        clean_text = clean_html_text(text)

        # Initialize the summarization pipeline with a smaller BART model
        # Use distilbart-cnn for CPU inference
        model_name = "sshleifer/distilbart-cnn-6-6"  # smaller model for CPU
        summarizer = pipeline("summarization", model=model_name)

        # BART has input token limits
        max_input_chars = 1024  # Conservative limit
        if len(clean_text) > max_input_chars:
            clean_text = clean_text[:max_input_chars]
            print(
                f"Text truncated to {max_input_chars} characters for BART model")

        # Generate summary with improved parameters for better ROUGE scores
        summary = summarizer(
            clean_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,  # Deterministic generation
            num_beams=4,      # Beam search for better quality
            early_stopping=True
        )[0]['summary_text']

        return summary

    except Exception as e:
        print(f"Error in BART summarization: {e}")
        # Fallback to extractive summary if generative fails
        print("Falling back to TextRank summarization")
        return textrank_summarization(text, 3)


def textrank_to_bart_summarization(text: str, num_textrank_sentences: int = 10,
                                   max_length: int = 100, min_length: int = 30) -> str:
    """
    Generate a hybrid summary using TextRank first, then BART

    Args:
        text: Input text to summarize
        num_textrank_sentences: Number of sentences for TextRank
        max_length: Maximum length of the final summary
        min_length: Minimum length of the final summary

    Returns:
        Hybrid summary
    """
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    try:
        # Extract more sentences with TextRank to capture more information
        textrank_summary = textrank_summarization(text, num_textrank_sentences)

        # Initialize BART
        model_name = "sshleifer/distilbart-cnn-6-6"  # smaller model for CPU
        summarizer = pipeline("summarization", model=model_name)

        # BART has input token limits
        max_input_chars = 1024  # Conservative limit
        if len(textrank_summary) > max_input_chars:
            textrank_summary = textrank_summary[:max_input_chars]

        # Configure BART to preserve more of the original wording
        summary = summarizer(
            textrank_summary,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,  # Use greedy decoding for more faithful summaries
            num_beams=4       # Beam search for better quality
        )[0]['summary_text']

        return summary

    except Exception as e:
        print(f"Error in TextRank→BART summarization: {e}")
        # Fallback to extractive summary if generative fails
        return textrank_summarization(text, 3)


def evaluate_summaries(original_text: str, summaries: Dict[str, str]) -> Dict[str, Dict]:
    """
    Evaluate summaries against original text using ROUGE metrics

    Args:
        original_text: Original article text
        summaries: Dictionary of summaries to evaluate

    Returns:
        Dictionary of ROUGE scores for each summary
    """
    rouge = Rouge()
    results = {}

    for name, summary in summaries.items():
        try:
            # Calculate ROUGE scores
            scores = rouge.get_scores(summary, original_text)

            # Store the results
            results[name] = {
                'rouge-1': scores[0]['rouge-1'],
                'rouge-2': scores[0]['rouge-2'],
                'rouge-l': scores[0]['rouge-l']
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores for {name}: {e}")
            results[name] = {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

    return results


def process_rss_feed(file_path: str = 'rss.xml', max_entries: int = 5,
                     textrank_sentences: int = 3, hybrid_textrank_sentences: int = 10) -> Dict:
    """
    Process RSS feed with summarization and ROUGE evaluation

    Args:
        file_path: Path to RSS feed XML file
        max_entries: Maximum number of entries to process
        textrank_sentences: Number of sentences for TextRank summarization
        hybrid_textrank_sentences: Number of sentences for the TextRank phase of hybrid summarization

    Returns:
        Dictionary with processing results
    """
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        # Prepare results container
        results = {
            "status": "success",
            "feed_title": feed.feed.get('title', 'No title'),
            "entries": []
        }

        # Process only a limited number of entries
        num_entries = min(max_entries, len(feed.entries))
        print(f"Processing first {num_entries} articles...")

        for i, entry in enumerate(feed.entries[:num_entries]):
            print(
                f"\n--- Processing article {i+1}: {entry.get('title', 'No title')} ---")

            # Extract content
            content = get_entry_content(entry)
            print(f"Content length: {len(content)} characters")

            # Generate summaries
            print("Generating TextRank summary...")
            textrank_summary = textrank_summarization(
                content, textrank_sentences)

            print("Generating BART summary...")
            bart_summary = bart_summarization(content)

            print("Generating TextRank→BART pipeline summary...")
            textrank_to_bart_summary = textrank_to_bart_summarization(
                content, hybrid_textrank_sentences)

            # Prepare summaries for evaluation
            summaries = {
                'TextRank': textrank_summary,
                'BART': bart_summary,
                'TextRank→BART': textrank_to_bart_summary
            }

            # Evaluate with ROUGE
            print("Evaluating summaries...")
            rouge_scores = evaluate_summaries(content, summaries)

            # Store entry result
            entry_result = {
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', ''),
                "published": entry.get('published', ''),
                "author": entry.get('author', 'Unknown'),
                "extractive_summary": textrank_summary,
                "abstractive_summary": bart_summary,
                "textrank_to_bart_summary": textrank_to_bart_summary,
                "rouge_scores": rouge_scores
            }

            results["entries"].append(entry_result)

            # Print comparison
            print("\n--- Summary Comparison ---")
            for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                print(f"\n{metric.upper()} F1 Scores:")
                for method, scores in rouge_scores.items():
                    print(f"  {method}: {scores[metric]['f']:.4f}")

        # Calculate average ROUGE scores across all entries
        print("\n--- Average ROUGE Scores ---")
        avg_scores = calculate_average_rouge_scores(results["entries"])
        results["average_scores"] = avg_scores

        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            print(f"\nAverage {metric.upper()} F1 Scores:")
            for method, scores in avg_scores.items():
                print(f"  {method}: {scores[metric]['f']:.4f}")

        return results

    except Exception as e:
        print(f"Error processing RSS: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def calculate_average_rouge_scores(entries: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate average ROUGE scores across all entries

    Args:
        entries: List of processed entries with ROUGE scores

    Returns:
        Dictionary with average ROUGE scores
    """
    methods = ['TextRank', 'BART', 'TextRank→BART']
    metrics = ['rouge-1', 'rouge-2', 'rouge-l']
    score_types = ['f', 'p', 'r']

    # Initialize counters
    sums = {}
    for method in methods:
        sums[method] = {}
        for metric in metrics:
            sums[method][metric] = {
                score_type: 0.0 for score_type in score_types}

    # Sum up all scores
    for entry in entries:
        rouge_scores = entry.get('rouge_scores', {})
        for method in methods:
            if method in rouge_scores:
                for metric in metrics:
                    for score_type in score_types:
                        sums[method][metric][score_type] += rouge_scores[method][metric][score_type]

    # Calculate averages
    avg_scores = {}
    num_entries = len(entries)
    if num_entries > 0:
        for method in methods:
            avg_scores[method] = {}
            for metric in metrics:
                avg_scores[method][metric] = {
                    score_type: sums[method][metric][score_type] / num_entries
                    for score_type in score_types
                }

    return avg_scores
