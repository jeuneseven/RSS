# summarizer.py
import traceback
from typing import Dict, List, Tuple, Any, Optional
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Import summarizer factory
from summarizer.summarizer_factory import SummarizerFactory

# Import for ROUGE evaluation
from rouge import Rouge

# Import common utilities
from utils import clean_html_text, get_entry_content


def process_rss_feed(
    file_path: str = 'index.xml',
    max_entries: int = 5,
    extractive_method: str = "textrank",
    abstractive_method: str = "bart",
    hybrid_method: str = "textrank-bart",
    extractive_sentences: int = 3,
    hybrid_extractive_sentences: int = 10,
    custom_summarizers: Dict[str, Any] = None
) -> Dict:
    """
    Process RSS feed with summarization and ROUGE evaluation supporting multiple algorithms

    Args:
        file_path: Path to RSS feed XML file
        max_entries: Maximum number of entries to process
        extractive_method: Name of extractive summarizer to use
        abstractive_method: Name of abstractive summarizer to use
        hybrid_method: Name of hybrid summarizer to use
        extractive_sentences: Number of sentences for extractive summarization
        hybrid_extractive_sentences: Number of sentences for the extractive phase of hybrid summarization
        custom_summarizers: Optional dict with custom summarizer instances to use

    Returns:
        Dictionary with processing results
    """
    try:
        import feedparser
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        # Initialize summarizers (from factory or custom provided)
        summarizers = {}

        if custom_summarizers and "extractive" in custom_summarizers:
            summarizers["extractive"] = custom_summarizers["extractive"]
            print(
                f"Using custom extractive summarizer: {summarizers['extractive'].get_metadata()['name']}")
        else:
            summarizers["extractive"] = SummarizerFactory.create_extractive_summarizer(
                extractive_method)
            print(f"Using extractive summarizer: {extractive_method}")

        if custom_summarizers and "abstractive" in custom_summarizers:
            summarizers["abstractive"] = custom_summarizers["abstractive"]
            print(
                f"Using custom abstractive summarizer: {summarizers['abstractive'].get_metadata()['name']}")
        else:
            summarizers["abstractive"] = SummarizerFactory.create_abstractive_summarizer(
                abstractive_method)
            print(f"Using abstractive summarizer: {abstractive_method}")

        if custom_summarizers and "hybrid" in custom_summarizers:
            summarizers["hybrid"] = custom_summarizers["hybrid"]
            print(
                f"Using custom hybrid summarizer: {summarizers['hybrid'].get_metadata()['name']}")
        else:
            # Check if it's a factory-provided hybrid or we need to create it
            try:
                summarizers["hybrid"] = SummarizerFactory.create_hybrid_summarizer(
                    hybrid_method)
            except ValueError:
                # Create a custom hybrid if standard one not found
                parts = hybrid_method.split('-')
                if len(parts) == 2:
                    ext = SummarizerFactory.create_extractive_summarizer(
                        parts[0])
                    abs = SummarizerFactory.create_abstractive_summarizer(
                        parts[1])

                    from summarizer.hybrid.base_hybrid import BaseHybridSummarizer
                    summarizers["hybrid"] = BaseHybridSummarizer(ext, abs)
                else:
                    raise ValueError(
                        f"Invalid hybrid method format: {hybrid_method}. Use format 'extractive-abstractive'")

            print(f"Using hybrid summarizer: {hybrid_method}")

        # Prepare results container
        results = {
            "status": "success",
            "feed_title": feed.feed.get('title', 'No title'),
            "entries": [],
            "summarizers_info": {
                "extractive": summarizers["extractive"].get_metadata(),
                "abstractive": summarizers["abstractive"].get_metadata(),
                "hybrid": summarizers["hybrid"].get_metadata()
            }
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
            print(f"Generating {extractive_method} summary...")
            extractive_summary = summarizers["extractive"].summarize(
                content, sentences_count=extractive_sentences)

            print(f"Generating {abstractive_method} summary...")
            abstractive_summary = summarizers["abstractive"].summarize(content)

            print(f"Generating hybrid {hybrid_method} summary...")
            hybrid_summary = summarizers["hybrid"].summarize(
                content, textrank_sentences=hybrid_extractive_sentences)

            # Prepare summaries for evaluation
            summaries = {
                extractive_method.capitalize(): extractive_summary,
                abstractive_method.capitalize(): abstractive_summary,
                hybrid_method.capitalize().replace('-', '→'): hybrid_summary
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
                "extractive_summary": extractive_summary,
                "abstractive_summary": abstractive_summary,
                "hybrid_summary": hybrid_summary,
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


def calculate_average_rouge_scores(entries: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate average ROUGE scores across all entries

    Args:
        entries: List of processed entries with ROUGE scores

    Returns:
        Dictionary with average ROUGE scores
    """
    # First entry to get method names
    if not entries:
        return {}

    first_entry = entries[0]
    rouge_scores = first_entry.get('rouge_scores', {})
    methods = list(rouge_scores.keys())
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
