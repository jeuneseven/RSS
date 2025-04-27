import re


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
