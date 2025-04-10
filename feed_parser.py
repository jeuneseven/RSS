import feedparser
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline
from rouge import Rouge


def clean_html_text(text):
    """Remove HTML tags and entities from text"""
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Replace HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def textrank_summarization(text, num_sentences=3):
    """
    Generate an extractive summary using TextRank algorithm from sumy library
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
        sentences = clean_text.split('. ')
        return '. '.join(sentences[:num_sentences]) + ('.' if not sentences[0].endswith('.') else '')


def bart_summarization(text, max_length=150, min_length=40):
    """
    Generate an abstractive summary using BART model from transformers library
    """
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    try:
        # Clean text
        clean_text = clean_html_text(text)

        # Initialize the summarization pipeline with BART
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # BART has input token limits
        max_input_chars = 1024  # Conservative limit
        if len(clean_text) > max_input_chars:
            clean_text = clean_text[:max_input_chars]
            print(
                f"Text truncated to {max_input_chars} characters for BART model")

        # Generate summary
        summary = summarizer(clean_text, max_length=max_length, min_length=min_length)[
            0]['summary_text']

        return summary

    except Exception as e:
        print(f"Error in BART summarization: {e}")
        # Fallback to extractive summary if generative fails
        return textrank_summarization(text, 2)


def evaluate_summaries(original_text, summaries):
    """
    Evaluate summaries against original text using ROUGE metrics
    Returns a dictionary of scores for each summary
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


def get_entry_content(entry):
    """Extract content from RSS entry trying various fields"""
    # Define priority order of fields to check
    content_fields = [
        ('description', lambda e: e.get('description')),
        ('content', lambda e: e.get('content', [{}])[0].get('value') if isinstance(
            e.get('content', []), list) else str(e.get('content', ''))),
        ('summary', lambda e: e.get('summary')),
        ('summary_detail', lambda e: e.get('summary_detail', {}).get('value'))
    ]

    # Try each field in order
    for field_name, getter in content_fields:
        content = getter(entry)
        if content and len(content.strip()) > 0:
            print(f"Found content in '{field_name}' field")
            return content

    # If no content found in standard fields, check all fields
    print("No standard content fields found, checking all fields for text content")
    for key in entry.keys():
        value = entry.get(key)
        if isinstance(value, str) and len(value) > 100:
            print(f"Found content in '{key}' field")
            return value

    # Return fallback message if no content found
    print("No suitable content field found")
    return "No content available to summarize."


def process_rss_feed(file_path='rss.xml'):
    """Process an RSS feed, summarize the first article, and evaluate with ROUGE"""
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        # Print basic information
        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        # Check if there are entries
        if len(feed.entries) == 0:
            print("No articles found.")
            return

        # Process only the first article
        entry = feed.entries[0]
        print("\nAnalyzing first article:")
        print(f"Title: {entry.get('title', 'No title')}")

        # Find content
        content = get_entry_content(entry)
        print(f"Original length: {len(content)} characters")
        # Show just the beginning for readability
        print(f"Original: {content[:300]}..." if len(
            content) > 300 else f"Original: {content}")

        # Generate summaries
        print("\nGenerating summaries...")
        textrank_summary = textrank_summarization(content, 3)
        bart_summary = bart_summarization(content)

        # Print summaries
        print(f"\nExtractive Summary (TextRank):")
        print(f"Length: {len(textrank_summary)} characters")
        print(f"Summary: {textrank_summary}")

        print(f"\nAbstractive Summary (BART):")
        print(f"Length: {len(bart_summary)} characters")
        print(f"Summary: {bart_summary}")

        # Evaluate summaries using ROUGE
        print("\nEvaluating summaries using ROUGE metrics...")
        summaries = {
            'TextRank': textrank_summary,
            'BART': bart_summary
        }

        # Since we don't have a human-written reference summary,
        # we'll use the original text as the reference
        # Note: This is not ideal, but provides a basis for comparison
        rouge_scores = evaluate_summaries(content, summaries)

        # Print ROUGE scores
        print("\n--- ROUGE Evaluation Results ---")
        for name, scores in rouge_scores.items():
            print(f"\n{name} Summary:")
            print(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
            print(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
            print(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")

            print(f"ROUGE-1 Precision: {scores['rouge-1']['p']:.4f}")
            print(f"ROUGE-2 Precision: {scores['rouge-2']['p']:.4f}")
            print(f"ROUGE-L Precision: {scores['rouge-l']['p']:.4f}")

            print(f"ROUGE-1 Recall: {scores['rouge-1']['r']:.4f}")
            print(f"ROUGE-2 Recall: {scores['rouge-2']['r']:.4f}")
            print(f"ROUGE-L Recall: {scores['rouge-l']['r']:.4f}")

        # Compare the summaries
        print("\n--- Summary Comparison ---")
        if rouge_scores['TextRank']['rouge-1']['f'] > rouge_scores['BART']['rouge-1']['f']:
            print("TextRank performed better on ROUGE-1 F1 score.")
        elif rouge_scores['TextRank']['rouge-1']['f'] < rouge_scores['BART']['rouge-1']['f']:
            print("BART performed better on ROUGE-1 F1 score.")
        else:
            print("TextRank and BART performed equally on ROUGE-1 F1 score.")

        if rouge_scores['TextRank']['rouge-2']['f'] > rouge_scores['BART']['rouge-2']['f']:
            print("TextRank performed better on ROUGE-2 F1 score.")
        elif rouge_scores['TextRank']['rouge-2']['f'] < rouge_scores['BART']['rouge-2']['f']:
            print("BART performed better on ROUGE-2 F1 score.")
        else:
            print("TextRank and BART performed equally on ROUGE-2 F1 score.")

        if rouge_scores['TextRank']['rouge-l']['f'] > rouge_scores['BART']['rouge-l']['f']:
            print("TextRank performed better on ROUGE-L F1 score.")
        elif rouge_scores['TextRank']['rouge-l']['f'] < rouge_scores['BART']['rouge-l']['f']:
            print("BART performed better on ROUGE-L F1 score.")
        else:
            print("TextRank and BART performed equally on ROUGE-L F1 score.")

        print("\nNOTE: Since we're using the original text as the reference, these scores")
        print("      reflect how well each summary retains information from the original.")
        print("      Extractive methods may have an advantage in this evaluation setting.")

    except Exception as e:
        print(f"Error processing RSS: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    process_rss_feed()
