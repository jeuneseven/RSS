import re


def clean_html_text(text):
    """
    Remove HTML tags and special entities from a given text.

    Args:
        text (str): Raw HTML text

    Returns:
        str: Cleaned plain text
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Replace common HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_entry_content(entry):
    """
    Try to extract main content from an RSS entry by checking common fields.

    Args:
        entry (dict): RSS feed entry

    Returns:
        str: Extracted content or fallback message
    """
    # Define content fields to check in order of preference
    content_fields = [
        ('description', lambda e: e.get('description')),
        ('content', lambda e: e.get('content', [{}])[0].get('value') if isinstance(
            e.get('content', []), list) else str(e.get('content', ''))),
        ('summary', lambda e: e.get('summary')),
        ('summary_detail', lambda e: e.get('summary_detail', {}).get('value'))
    ]

    # Try to extract content from standard fields
    for field_name, getter in content_fields:
        content = getter(entry)
        if content and len(content.strip()) > 0:
            print(f"Found content in '{field_name}' field")
            return clean_html_text(content)

    # If no standard fields found, check all fields for text content
    print("No standard content fields found, checking all fields for text content")
    for key in entry.keys():
        value = entry.get(key)
        if isinstance(value, str) and len(value) > 100:
            print(f"Found content in '{key}' field")
            return value

    print("No suitable content field found")
    return "No content available to summarize."


def split_into_sentences(text):
    """
    Split a block of text into individual sentences using basic punctuation rules.

    Args:
        text (str): Input paragraph or article

    Returns:
        list: List of sentence strings
    """
    # Add markers for sentence endings
    text = re.sub(r'([.!?])', r'\1<SPLIT>', text)

    # Split on markers
    parts = text.split('<SPLIT>')

    # Clean and filter sentences
    sentences = [s.strip() for s in parts if len(s.strip()) > 0]

    return sentences


def format_results_for_display(results):
    """
    Format processing results for better display

    Args:
        results (dict): Raw processing results

    Returns:
        str: Formatted string for display
    """
    output = []
    output.append("=== RSS Feed Processing Results ===\n")

    if results.get("status") != "success":
        output.append(f"Error: {results.get('message', 'Unknown error')}")
        return "\n".join(output)

    output.append(f"Feed Title: {results.get('feed_title', 'Unknown')}")
    output.append(f"Total Entries: {len(results.get('entries', []))}\n")

    for i, entry in enumerate(results.get('entries', [])[:5]):  # Show first 5 entries
        output.append(f"=== Article {i+1} ===")
        output.append(f"Title: {entry.get('title', 'No title')}")
        output.append(f"Link: {entry.get('link', 'No link')}")
        output.append(f"Published: {entry.get('published', 'Unknown')}")

        # Classification results
        if 'classification' in entry:
            cls = entry['classification']
            if 'error' not in cls:
                output.append(
                    f"Category: {cls['top_category']} (Confidence: {cls['confidence']:.2f})")

        # Summary results
        if 'extractive_summary' in entry:
            output.append(
                f"\nExtractive Summary: {entry['extractive_summary'][:200]}...")

        if 'abstractive_summary' in entry:
            output.append(
                f"Abstractive Summary: {entry['abstractive_summary'][:200]}...")

        if 'textrank_to_bart_summary' in entry:
            output.append(
                f"TextRank→BART Summary: {entry['textrank_to_bart_summary'][:200]}...")

        # ROUGE scores
        if 'rouge_scores' in entry:
            output.append("\nROUGE Scores:")
            for method, scores in entry['rouge_scores'].items():
                output.append(
                    f"  {method}: ROUGE-1={scores['rouge-1']['f']:.3f}")

        output.append("")  # Empty line between entries

    if len(results.get('entries', [])) > 5:
        output.append(f"... and {len(results['entries']) - 5} more entries")

    return "\n".join(output)
