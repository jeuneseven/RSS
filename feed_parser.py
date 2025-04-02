import feedparser
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
from collections import Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


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
    Generate an extractive summary using TextRank algorithm
    """
    # Check if text is empty
    if not text or len(text.strip()) == 0:
        return "No content available to summarize."

    # Clean text (remove HTML tags)
    clean_text = clean_html_text(text)

    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")

        # Process the text with spaCy
        doc = nlp(clean_text)

        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]

        # Check if we have enough sentences to summarize
        if len(sentences) <= num_sentences:
            return clean_text

        # Create sentence embeddings
        sentence_vectors = []
        for sent in sentences:
            # Process each sentence to get its vector
            sent_doc = nlp(sent)
            # Skip empty sentences
            if len(sent_doc) == 0:
                # Default vector dimension for en_core_web_sm
                sentence_vectors.append(np.zeros((96,)))
                continue

            # Get average word vector for the sentence
            vec = np.mean(
                [word.vector for word in sent_doc if not word.is_stop and not word.is_punct], axis=0)
            sentence_vectors.append(vec)

        # Create similarity matrix
        sim_matrix = np.zeros([len(sentences), len(sentences)])

        # Calculate similarity between sentence vectors
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    # Handle zero vectors (sentences with only stop words)
                    if np.all(sentence_vectors[i] == 0) or np.all(sentence_vectors[j] == 0):
                        sim_matrix[i][j] = 0
                    else:
                        # Reshape vectors for cosine_similarity which expects 2D arrays
                        vec_i = sentence_vectors[i].reshape(1, -1)
                        vec_j = sentence_vectors[j].reshape(1, -1)
                        sim_matrix[i][j] = cosine_similarity(vec_i, vec_j)[
                            0, 0]

        # Apply PageRank algorithm
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)

        # Add position bias - give higher weight to sentences at the beginning
        for i in range(len(sentences)):
            # Decreasing weight for later sentences
            position_weight = 1 / (1 + 0.1 * i)
            scores[i] = scores[i] * position_weight

        # Get top-ranked sentences
        ranked_sentences = sorted(
            ((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)

        # Select top N sentences and sort them by position in the original text
        top_sentence_indices = sorted(
            [idx for _, idx, _ in ranked_sentences[:num_sentences]])

        # Build summary by joining selected sentences
        summary = ' '.join(sentences[i] for i in top_sentence_indices)

        return summary

    except Exception as e:
        print(f"Error in TextRank summarization: {e}")
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
            summary = textrank_summarization(
                content, 2)  # Get 2 key sentences
            print(f"Original length: {len(content)} characters")
            print(f"Original: {content}")
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
