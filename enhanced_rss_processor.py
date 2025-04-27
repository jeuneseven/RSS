import feedparser
import argparse
import json
import traceback

# Import summarization functions from feed_parser_summary
from feed_parser_summary import textrank_summarization, bart_summarization, evaluate_summaries

# Import classification functions from bert_classifier
from bert_classifier import BERTTextClassifier, classify_rss_entries, process_rss_feed_with_classification

# Import common utilities
from utils import get_entry_content


def process_rss_feed_enhanced(file_path='rss.xml', summarize=True, classify=True, custom_labels=None):
    """
    Enhanced RSS feed processor that includes both summarization and classification

    Args:
        file_path (str): Path to the RSS feed file
        summarize (bool): Whether to generate summaries
        classify (bool): Whether to classify the entries
        custom_labels (list): Optional list of custom labels for classification

    Returns:
        dict: Processing results
    """
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        # Print basic information
        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        # Check if there are entries
        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        results = {
            "status": "success",
            "feed_title": feed.feed.get('title', 'No title'),
            "entries": []
        }

        # Initialize classifier if needed
        classifier = None
        if classify:
            print("\nInitializing BERT classifier...")
            labels = custom_labels or [
                "Technology", "Business", "Science", "Entertainment", "Health"]
            classifier = BERTTextClassifier(
                num_labels=len(labels), labels=labels)

        # Process each entry
        for i, entry in enumerate(feed.entries):
            print(
                f"\n--- Processing article {i+1}: {entry.get('title', 'No title')} ---")

            # Get content
            content = get_entry_content(entry)

            entry_result = {
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', ''),
                "published": entry.get('published', ''),
                "author": entry.get('author', 'Unknown')
            }

            # Generate summaries if requested
            if summarize:
                print("Generating summaries...")

                textrank_summary = textrank_summarization(content, 3)
                entry_result["extractive_summary"] = textrank_summary
                print(
                    f"Extractive summary length: {len(textrank_summary)} chars")

                try:
                    bart_summary = bart_summarization(content)
                    entry_result["abstractive_summary"] = bart_summary
                    print(
                        f"Abstractive summary length: {len(bart_summary)} chars")
                except Exception as e:
                    print(f"Error generating abstractive summary: {e}")
                    entry_result["abstractive_summary"] = "Error generating summary"

            # Classify content if requested
            if classify and classifier:
                print("Classifying content...")
                try:
                    classification = classifier.classify(content)
                    entry_result["classification"] = classification
                    print(
                        f"Top category: {classification['top_category']} (Confidence: {classification['confidence']:.4f})")
                except Exception as e:
                    print(f"Error classifying content: {e}")
                    entry_result["classification"] = {"error": str(e)}

            # Add processed entry to results
            results["entries"].append(entry_result)

            # Limit output in verbose mode for clarity
            if i >= 4:  # Only show details for first 5 articles
                print(
                    f"\n... ({len(feed.entries) - 5} more articles processed)")
                break

        return results

    except Exception as e:
        print(f"Error processing RSS: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def save_results_to_file(results, output_file='processed_feed.json'):
    """Save processing results to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main function to run the enhanced RSS processor"""
    parser = argparse.ArgumentParser(
        description='Process RSS feeds with summarization and classification')
    parser.add_argument('--feed', type=str, default='rss.xml',
                        help='Path to RSS feed file')
    parser.add_argument('--no-summarize', action='store_true',
                        help='Skip summary generation')
    parser.add_argument('--no-classify', action='store_true',
                        help='Skip classification')
    parser.add_argument('--custom-labels', type=str,
                        help='Comma-separated list of custom labels for classification')
    parser.add_argument('--output', type=str,
                        default='processed_feed.json', help='Output file path')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only run summarization (old behavior)')
    parser.add_argument('--classify-only', action='store_true',
                        help='Only run classification')

    args = parser.parse_args()

    # Process custom labels if provided
    custom_labels = None
    if args.custom_labels:
        custom_labels = [label.strip()
                         for label in args.custom_labels.split(',')]
        print(f"Using custom labels: {custom_labels}")

    # Handle special modes
    if args.summary_only:
        # Use the original feed_parser_summary functionality
        from feed_parser_summary import process_rss_feed
        process_rss_feed(args.feed)
        return

    if args.classify_only:
        # Use the classification-only functionality
        results = process_rss_feed_with_classification(
            file_path=args.feed,
            classify=True,
            custom_labels=custom_labels
        )
        save_results_to_file(results, args.output)
        return

    # Process the feed with enhanced functionality
    results = process_rss_feed_enhanced(
        file_path=args.feed,
        summarize=not args.no_summarize,
        classify=not args.no_classify,
        custom_labels=custom_labels
    )

    # Save results
    save_results_to_file(results, args.output)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
