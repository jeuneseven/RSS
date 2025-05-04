import argparse
import json
import traceback
from typing import Dict, Any, List, Optional

# Import core functionalities
from summarizer import process_rss_feed, calculate_average_rouge_scores
from classifier import BERTTextClassifier
from utils import get_entry_content, clean_html_text

# Ensure NLTK Punkt tokenizer is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


def process_rss_feed_enhanced(
    file_path: str = 'rss.xml',
    max_entries: int = 5,
    textrank_sentences: int = 3,
    hybrid_textrank_sentences: int = 10,
    summarize: bool = True,
    classify: bool = True,
    custom_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Enhanced RSS feed processor that combines both summarization and classification

    Args:
        file_path (str): Path to the RSS feed file
        max_entries (int): Maximum number of entries to process
        textrank_sentences (int): Number of sentences for TextRank summarization
        hybrid_textrank_sentences (int): Number of sentences for the TextRank phase in hybrid summarization
        summarize (bool): Whether to generate summaries
        classify (bool): Whether to classify the entries
        custom_labels (list): Optional list of custom labels for classification

    Returns:
        dict: Processing results
    """
    try:
        print(f"RSS Feed Processor")
        print(f"=================")
        print(f"Feed file: {file_path}")
        print(f"Maximum entries: {max_entries}")
        print(f"TextRank sentences: {textrank_sentences}")
        print(f"Hybrid TextRank sentences: {hybrid_textrank_sentences}")
        print(f"Summarize: {summarize}")
        print(f"Classify: {classify}")
        if custom_labels:
            print(f"Custom labels: {custom_labels}")
        print(f"=================\n")

        # Process the feed with summarization functionality
        results = process_rss_feed(
            file_path=file_path,
            max_entries=max_entries,
            textrank_sentences=textrank_sentences,
            hybrid_textrank_sentences=hybrid_textrank_sentences
        )

        # Skip if there's an error
        if results.get("status") != "success":
            return results

        # Add classification if requested
        if classify:
            print("\nPerforming classification on entries...")
            labels = custom_labels or [
                "Technology", "Business", "Science", "Entertainment", "Health"]
            classifier = BERTTextClassifier(
                num_labels=len(labels), labels=labels)

            for entry in results["entries"]:
                # Extract content from entry
                title = entry.get("title", "")
                # Use summary if available to save time
                content = entry.get("extractive_summary", "")
                if not content:
                    # Re-create the content if not available (shouldn't happen normally)
                    print(
                        f"Warning: No extractive summary found for '{title}', using full content")
                    import feedparser
                    feed = feedparser.parse(file_path)
                    for feed_entry in feed.entries:
                        if feed_entry.get("title") == title:
                            content = get_entry_content(feed_entry)
                            break

                # Classify the content
                try:
                    classification = classifier.classify(content)
                    entry["classification"] = classification
                    print(f"Classified '{title}' as {classification['top_category']} "
                          f"(Confidence: {classification['confidence']:.4f})")
                except Exception as e:
                    print(f"Error classifying '{title}': {e}")
                    entry["classification"] = {"error": str(e)}

        return results

    except Exception as e:
        print(f"Error in enhanced processing: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def save_results_to_file(results: Dict[str, Any], output_file: str = 'processed_feed.json') -> None:
    """
    Save processing results to a JSON file

    Args:
        results: Processing results
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """
    Main entry point for the RSS summarizer and classifier application
    Handles command line arguments and executes the processing pipeline
    """
    parser = argparse.ArgumentParser(
        description='Process RSS feeds with multiple summarization techniques and optional classification'
    )

    # Add command line arguments
    parser.add_argument(
        '--feed',
        type=str,
        default='rss.xml',
        help='Path to RSS feed file (default: rss.xml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='processed_feed.json',
        help='Output file path (default: processed_feed.json)'
    )

    parser.add_argument(
        '--max-entries',
        type=int,
        default=5,
        help='Maximum number of entries to process (default: 5)'
    )

    parser.add_argument(
        '--textrank-sentences',
        type=int,
        default=3,
        help='Number of sentences for TextRank summarization (default: 3)'
    )

    parser.add_argument(
        '--hybrid-sentences',
        type=int,
        default=10,
        help='Number of sentences for TextRank phase in hybrid summarization (default: 10)'
    )

    parser.add_argument(
        '--no-summarize',
        action='store_true',
        help='Skip summary generation'
    )

    parser.add_argument(
        '--no-classify',
        action='store_true',
        help='Skip classification'
    )

    parser.add_argument(
        '--custom-labels',
        type=str,
        help='Comma-separated list of custom labels for classification (e.g., "Sports,Politics,Health")'
    )

    # Parse arguments
    args = parser.parse_args()

    # Process custom labels if provided
    custom_labels = None
    if args.custom_labels:
        custom_labels = [label.strip()
                         for label in args.custom_labels.split(',')]
        print(f"Using custom classification labels: {custom_labels}")

    # Process the feed with enhanced functionality
    results = process_rss_feed_enhanced(
        file_path=args.feed,
        max_entries=args.max_entries,
        textrank_sentences=args.textrank_sentences,
        hybrid_textrank_sentences=args.hybrid_sentences,
        summarize=not args.no_summarize,
        classify=not args.no_classify,
        custom_labels=custom_labels
    )

    # Save results
    save_results_to_file(results, args.output)

    # Print final comparison if successful
    if results.get("status") == "success":
        print("\n=== Summary of Average ROUGE Scores ===")
        avg_scores = results.get("average_scores", {})

        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            print(f"\nAverage {metric.upper()} F1 Scores:")
            for method, scores in avg_scores.items():
                if method in scores and metric in scores[method]:
                    print(f"  {method}: {scores[method][metric]['f']:.4f}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
