#!/usr/bin/env python3
# main.py
"""
RSS Summarizer and Classifier - Main Entry Point
Author: Your Name
Description: Process RSS feeds with summarization and classification
"""

import argparse
import json
import sys
import os
import feedparser
from typing import Dict, Any

# Import core modules that exist
from feed_parser import process_rss_feed as simple_process_rss_feed
from classifier import BERTTextClassifier
from utils import format_results_for_display, get_entry_content, clean_html_text
import config


def process_feed_with_classification(file_path: str, max_entries: int = 5, enable_classification: bool = False, categories: list = None, threshold: float = 0.3):
    """
    Process RSS feed with both summarization and classification
    Uses the existing feed_parser.py functionality
    """
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        if len(feed.entries) == 0:
            return {"status": "error", "message": "No articles found"}

        # Initialize classifier if needed
        classifier = None
        if enable_classification:
            print("Initializing text classifier...")
            classifier = BERTTextClassifier(
                num_labels=len(categories or config.DEFAULT_CATEGORIES),
                labels=categories or config.DEFAULT_CATEGORIES
            )

        # Prepare results
        results = {
            "status": "success",
            "feed_title": feed.feed.get('title', 'No title'),
            "entries": []
        }

        # Process entries
        num_entries = min(max_entries, len(feed.entries))
        print(f"Processing first {num_entries} articles...")

        for i, entry in enumerate(feed.entries[:num_entries]):
            print(
                f"\n--- Processing article {i+1}: {entry.get('title', 'No title')} ---")

            # Extract content
            content = get_entry_content(entry)
            print(f"Content length: {len(content)} characters")

            # Generate summaries using existing functions
            from feed_parser import textrank_summarization, bart_summarization, bart_summarization_from_textrank

            print("Generating summaries...")
            textrank_summary = textrank_summarization(content, 3)
            bart_summary = bart_summarization(content)
            hybrid_summary = bart_summarization_from_textrank(content)

            # Prepare entry result
            entry_result = {
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', ''),
                "published": entry.get('published', ''),
                "author": entry.get('author', 'Unknown'),
                "original_content": content[:500] + "..." if len(content) > 500 else content,
                "textrank_summary": textrank_summary,
                "bart_summary": bart_summary,
                "hybrid_summary": hybrid_summary
            }

            # Add classification if enabled
            if classifier:
                print("Classifying content...")
                classification = classifier.classify(
                    content, threshold=threshold)
                entry_result["classification"] = classification

            # Evaluate summaries
            summaries = {
                'TextRank': textrank_summary,
                'BART': bart_summary,
                'TextRank→BART': hybrid_summary
            }

            from feed_parser import evaluate_summaries
            rouge_scores = evaluate_summaries(content, summaries)
            entry_result["rouge_scores"] = rouge_scores

            results["entries"].append(entry_result)

            # Print summary info
            print(f"TextRank summary length: {len(textrank_summary)} chars")
            print(f"BART summary length: {len(bart_summary)} chars")
            print(f"Hybrid summary length: {len(hybrid_summary)} chars")

            if classifier and "classification" in entry_result:
                cls = entry_result["classification"]
                if "error" not in cls:
                    print(
                        f"Classification: {cls['top_category']} (confidence: {cls['confidence']:.3f})")

        return results

    except Exception as e:
        print(f"Error processing RSS: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def format_simple_results(results: Dict[str, Any]) -> str:
    """Format results for display"""
    if results.get("status") != "success":
        return f"Error: {results.get('message', 'Unknown error')}"

    output = []
    output.append("=" * 80)
    output.append("RSS FEED PROCESSING RESULTS")
    output.append("=" * 80)
    output.append(f"Feed: {results.get('feed_title', 'Unknown')}")
    output.append(f"Processed: {len(results.get('entries', []))} articles")
    output.append("")

    for i, entry in enumerate(results.get('entries', [])):
        output.append(f"{'='*20} ARTICLE {i+1} {'='*20}")
        output.append(f"Title: {entry.get('title', 'No title')}")
        output.append(f"Link: {entry.get('link', 'No link')}")
        output.append(f"Published: {entry.get('published', 'Unknown')}")
        output.append("")

        # Show original content preview
        output.append("Original Content (preview):")
        output.append(entry.get('original_content',
                      'No content')[:300] + "...")
        output.append("")

        # Show summaries
        output.append("SUMMARIES:")
        output.append("-" * 40)

        if 'textrank_summary' in entry:
            output.append("TextRank (Extractive):")
            output.append(f"  {entry['textrank_summary']}")
            output.append("")

        if 'bart_summary' in entry:
            output.append("BART (Abstractive):")
            output.append(f"  {entry['bart_summary']}")
            output.append("")

        if 'hybrid_summary' in entry:
            output.append("TextRank→BART (Hybrid):")
            output.append(f"  {entry['hybrid_summary']}")
            output.append("")

        # Show classification
        if 'classification' in entry:
            cls = entry['classification']
            if 'error' not in cls:
                output.append("CLASSIFICATION:")
                output.append(f"  Category: {cls['top_category']}")
                output.append(f"  Confidence: {cls['confidence']:.3f}")
                if len(cls['predictions']) > 1:
                    output.append("  All predictions:")
                    for pred in cls['predictions']:
                        output.append(
                            f"    {pred['label']}: {pred['confidence']:.3f}")
                output.append("")

        # Show ROUGE scores
        if 'rouge_scores' in entry:
            output.append("ROUGE SCORES (F1):")
            rouge_scores = entry['rouge_scores']
            for method, scores in rouge_scores.items():
                output.append(f"  {method}:")
                output.append(f"    ROUGE-1: {scores['rouge-1']['f']:.4f}")
                output.append(f"    ROUGE-2: {scores['rouge-2']['f']:.4f}")
                output.append(f"    ROUGE-L: {scores['rouge-l']['f']:.4f}")
            output.append("")

        output.append("")

    return "\n".join(output)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RSS Feed Summarizer and Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Process default RSS file
  python main.py -f news.xml -n 10                 # Process 10 entries from news.xml
  python main.py --classify                        # Enable classification
  python main.py --classify --categories Tech Business Health  # Custom categories
  python main.py -o results.json                   # Save results to JSON file
        """
    )

    # Input/Output options
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=config.DEFAULT_RSS_PATH,
        help=f"RSS XML file path (default: {config.DEFAULT_RSS_PATH})"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file to save results (optional)"
    )

    parser.add_argument(
        "-n", "--max-entries",
        type=int,
        default=config.MAX_ENTRIES,
        help=f"Maximum number of entries to process (default: {config.MAX_ENTRIES})"
    )

    # Classification options
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Enable text classification"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=config.DEFAULT_CATEGORIES,
        help=f"Classification categories (default: {' '.join(config.DEFAULT_CATEGORIES)})"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CLASSIFICATION_THRESHOLD,
        help=f"Classification confidence threshold (default: {config.CLASSIFICATION_THRESHOLD})"
    )

    # Display options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    # Simple mode (use original feed_parser.py only)
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple mode (original feed_parser.py functionality only)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.file):
        print(f"Error: RSS file '{args.file}' not found.")
        print(f"Please create an RSS XML file or specify an existing one with -f")
        sys.exit(1)

    print("=" * 60)
    print("RSS Summarizer and Classifier")
    print("=" * 60)
    print(f"Processing file: {args.file}")
    print(f"Max entries: {args.max_entries}")

    if args.classify:
        print(f"Classification: Enabled")
        print(f"Categories: {', '.join(args.categories)}")
        print(f"Threshold: {args.threshold}")
    else:
        print("Classification: Disabled")

    print("-" * 60)

    try:
        if args.simple:
            # Use original simple function
            print("Running in simple mode...")
            simple_process_rss_feed(args.file)
        else:
            # Use enhanced function with classification
            results = process_feed_with_classification(
                file_path=args.file,
                max_entries=args.max_entries,
                enable_classification=args.classify,
                categories=args.categories,
                threshold=args.threshold
            )

            if results["status"] != "success":
                print(f"Error: {results.get('message', 'Unknown error')}")
                sys.exit(1)

            # Display results
            if args.format == "json":
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                formatted_output = format_simple_results(results)
                print(formatted_output)

            # Save to file if requested
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {args.output}")

            print(f"\nProcessing completed successfully!")
            print(f"Processed {len(results['entries'])} articles.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_interactive():
    """Interactive mode for easier usage"""
    print("=" * 60)
    print("RSS Summarizer - Interactive Mode")
    print("=" * 60)

    # Get RSS file
    rss_file = input(
        f"RSS file path (default: {config.DEFAULT_RSS_PATH}): ").strip()
    if not rss_file:
        rss_file = config.DEFAULT_RSS_PATH

    if not os.path.exists(rss_file):
        print(f"Error: File '{rss_file}' not found.")
        print("Please make sure you have an RSS XML file in the current directory.")
        return

    # Get number of entries
    max_entries_input = input(
        f"Maximum entries to process (default: {config.MAX_ENTRIES}): ").strip()
    try:
        max_entries = int(
            max_entries_input) if max_entries_input else config.MAX_ENTRIES
    except ValueError:
        max_entries = config.MAX_ENTRIES

    # Ask about classification
    classify = input(
        "Enable classification? (y/N): ").strip().lower().startswith('y')

    print("\nProcessing...")

    try:
        # Process with default settings
        results = process_feed_with_classification(
            file_path=rss_file,
            max_entries=max_entries,
            enable_classification=classify
        )

        if results["status"] != "success":
            print(f"Error: {results.get('message', 'Unknown error')}")
            return

        # Display results
        formatted_output = format_simple_results(results)
        print("\n" + formatted_output)

        # Ask to save
        save = input(
            "\nSave results to file? (y/N): ").strip().lower().startswith('y')
        if save:
            output_file = input(
                f"Output file (default: {config.DEFAULT_OUTPUT_PATH}): ").strip()
            if not output_file:
                output_file = config.DEFAULT_OUTPUT_PATH

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        run_interactive()
    else:
        # Arguments provided, run command line mode
        main()
