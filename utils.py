import re


def clean_html_text(text):
    """
    Remove HTML tags and special entities from a given text.

    Args:
        text (str): Raw HTML text

    Returns:
        str: Cleaned plain text
    """
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def get_entry_content(entry):
    """
    Try to extract main content from an RSS entry by checking common fields.

    Args:
        entry (dict): RSS feed entry

    Returns:
        str: Extracted content or fallback message
    """
    content_fields = [
        ('description', lambda e: e.get('description')),
        ('content', lambda e: e.get('content', [{}])[0].get('value') if isinstance(
            e.get('content', []), list) else str(e.get('content', ''))),
        ('summary', lambda e: e.get('summary')),
        ('summary_detail', lambda e: e.get('summary_detail', {}).get('value'))
    ]

    for field_name, getter in content_fields:
        content = getter(entry)
        if content and len(content.strip()) > 0:
            print(f"Found content in '{field_name}' field")
            return clean_html_text(content)

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
    text = re.sub(r'([.!?])', r'\1<SPLIT>', text)
    parts = text.split('<SPLIT>')
    sentences = [s.strip() for s in parts if len(s.strip()) > 0]
    return sentences
