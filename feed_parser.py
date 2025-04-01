import feedparser
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
from collections import Counter


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


def extractive_summarization(text, num_sentences=3):
    """
    Generate an extractive summary using spaCy's linguistic features
    """
    # Check if text is empty
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    # Clean text (remove HTML tags)
    clean_text = clean_html_text(text)

    try:
        # Load spaCy model - using small model for efficiency
        nlp = spacy.load("en_core_web_sm")

        # Process the text with spaCy
        doc = nlp(clean_text)

        # Tokenize the text into sentences using spaCy
        sentences = [sent.text.strip() for sent in doc.sents]

        # Check if we have enough sentences to summarize
        if len(sentences) <= num_sentences:
            return clean_text

        # Calculate word frequencies with proper lemmatization
        word_frequencies = Counter()
        for word in doc:
            if not word.is_stop and not word.is_punct and not word.is_space:
                # Use lemma to normalize words
                word_frequencies[word.lemma_] += 1

        # Normalize word frequencies
        max_frequency = max(word_frequencies.values(), default=1)
        normalized_frequencies = {
            word: freq/max_frequency for word, freq in word_frequencies.items()}

        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(doc.sents):
            for word in sentence:
                if word.lemma_ in normalized_frequencies:
                    # Add additional weight to named entities
                    entity_weight = 1.5 if word.ent_type_ else 1.0
                    # Add weight for being in the first sentence
                    position_weight = 1.2 if i == 0 else 1.0

                    if i not in sentence_scores:
                        sentence_scores[i] = 0

                    sentence_scores[i] += normalized_frequencies[word.lemma_] * \
                        entity_weight * position_weight

            # Normalize by sentence length
            if i in sentence_scores and len(sentence) > 0:
                sentence_scores[i] = sentence_scores[i] / len(sentence)

        # Select top sentences
        ranked_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentence_indices = [idx for idx,
                                _ in ranked_sentences[:num_sentences]]
        top_sentence_indices.sort()  # Sort by position in original text

        # Combine selected sentences
        summary = ' '.join(sentences[i] for i in top_sentence_indices)

        return summary
    except Exception as e:
        print(f"Error in summarization: {e}")
        # Fallback to first few sentences if processing fails
        return ' '.join(sentences[:num_sentences]) if 'sentences' in locals() else "Error generating summary."


def process_rss_feed(file_path='rss.xml'):
    """Process an RSS feed and generate summaries for articles"""
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

        print("\nArticles with summaries:")
        for i, entry in enumerate(feed.entries[:3]):  # Process first 3 entries
            print(f"\nArticle {i+1}:")
            print(f"Title: {entry.get('title', 'No title')}")

            # Find content using a more robust approach
            content = get_entry_content(entry)

            # Generate and print summary
            summary = extractive_summarization(
                content, 2)  # Get 2 key sentences
            print(f"Original length: {len(content)} characters")
            print(f"Summary length: {len(summary)} characters")
            print(f"Summary: {summary}")
            print("-" * 50)
    except Exception as e:
        print(f"Error parsing RSS: {e}")
        import traceback
        traceback.print_exc()


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


if __name__ == "__main__":
    process_rss_feed()
