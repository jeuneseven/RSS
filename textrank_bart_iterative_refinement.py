import feedparser
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge import Rouge
from bert_score import score as bert_score
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import re
import time

# Download necessary NLTK data
nltk.download('punkt')

# Set device for model inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize BART model & tokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn').to(device)

# Initialize ROUGE evaluator
rouge_evaluator = Rouge()


def clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = BeautifulSoup(html or "", "html.parser").get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()


def fetch_rss_entries(rss_url: str, max_articles: int = 5):
    """Fetch RSS feed entries (up to max_articles)."""
    feed = feedparser.parse(rss_url)
    return feed.entries[:max_articles]


def extract_textrank(text: str, ratio: float = None, sentences_count: int = None) -> str:
    """
    Extract key sentences using TextRank from Sumy.
    Supports ratio or fixed sentences_count.
    """
    sents = sent_tokenize(text)
    if not sents:
        return text

    if sentences_count is None:
        if ratio is not None:
            sentences_count = max(1, int(len(sents) * ratio))
        else:
            sentences_count = 3

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sents = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary_sents)


def generate_bart(text: str, min_len: int = 40, max_len: int = 150) -> str:
    """Generate abstractive summary using BART."""
    if not text.strip():
        return ""

    # Handle input length - BART handles up to 1024 tokens
    inputs = bart_tokenizer(text, truncation=True,
                            max_length=1024, return_tensors="pt").to(device)

    # Generate summary
    summary_ids = bart_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=4,
        length_penalty=2.0,
        max_length=max_len,
        min_length=min_len,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def original_iterative_refinement(text: str, iterations: int = 2, extract_ratio: float = 0.2) -> str:
    """
    Original iterative refinement method.
    """
    current = text
    for _ in range(iterations):
        extracted = extract_textrank(current, ratio=extract_ratio)
        current = generate_bart(extracted)
    return current


def extract_diverse_sentences(text: str, ratio: float = 0.3, diversity_penalty: float = 0.5) -> str:
    """
    Extract diverse key sentences using TextRank with diversity penalty.

    Args:
        text: Input text
        ratio: Extraction ratio
        diversity_penalty: Penalty for similar sentences (higher = more diverse)

    Returns:
        Diverse extractive summary
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return text

    # If very few sentences, return all
    if len(sentences) <= 3:
        return text

    # Calculate number of sentences to extract
    count = max(1, int(len(sentences) * ratio))

    # Get TextRank sentences in ranked order
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sents = summarizer(
        parser.document, count * 2)  # Get more than needed

    # Convert to strings
    candidates = [str(s) for s in summary_sents]

    # Select diverse subset
    selected = [candidates[0]]  # Start with highest ranked

    # Function to calculate similarity between sentences
    def sentence_similarity(s1, s2):
        # Simple word overlap for demonstration
        # In production, use embeddings or better similarity measures
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    # Select remaining sentences with diversity penalty
    for candidate in candidates[1:]:
        # Calculate max similarity to already selected sentences
        max_sim = max([sentence_similarity(candidate, s)
                      for s in selected]) if selected else 0

        # Apply diversity penalty
        if max_sim < diversity_penalty and len(selected) < count:
            selected.append(candidate)

        # Stop if we have enough
        if len(selected) >= count:
            break

    return " ".join(selected)


def evaluate_rouge(summary: str, reference: str) -> dict:
    """Compute ROUGE scores."""
    if not summary or not reference:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

    try:
        scores = rouge_evaluator.get_scores(summary, reference)[0]
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}


def evaluate_bertscore(summary: str, reference: str) -> float:
    """Compute BERTScore F1."""
    if not summary or not reference:
        return 0.0

    try:
        P, R, F1 = bert_score([summary], [reference], lang='en')
        return F1.item()
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return 0.0


def evaluate_all_metrics(summaries: dict, reference: str) -> dict:
    """
    Evaluate all summaries against reference text.

    Args:
        summaries: Dictionary of {method_name: summary_text}
        reference: Reference text

    Returns:
        Dictionary of metrics by method
    """
    results = []

    for method, summary in summaries.items():
        rouge_scores = evaluate_rouge(summary, reference)
        bert_s = evaluate_bertscore(summary, reference)

        results.append({
            'method': method,
            'rouge-1': rouge_scores['rouge-1'],
            'rouge-2': rouge_scores['rouge-2'],
            'rouge-l': rouge_scores['rouge-l'],
            'bertscore': bert_s,
            'length': len(summary.split())
        })

    return results


def adaptive_parameters(text: str) -> dict:
    """
    Determine optimal summarization parameters based on text characteristics.

    Args:
        text: Input text

    Returns:
        Dictionary of parameters
    """
    word_count = len(text.split())
    sentence_count = len(sent_tokenize(text))

    # Default parameters
    params = {
        'iterations': 2,
        'initial_ratio': 0.3,
        'min_length': 100,
        'preserve_original': True,
        'decay_factor': 0.7
    }

    # Adjust based on text length
    if word_count < 200:  # Short text
        params['iterations'] = 1
        params['initial_ratio'] = 0.5
    elif word_count > 1000:  # Long text
        params['iterations'] = 3
        params['initial_ratio'] = 0.25
        params['decay_factor'] = 0.6

    # Adjust based on sentence complexity
    avg_sent_length = word_count / max(1, sentence_count)
    if avg_sent_length > 25:  # Complex sentences
        params['initial_ratio'] = min(0.4, params['initial_ratio'] + 0.1)

    return params


def improved_iterative_refinement(text: str, iterations: int = 2, initial_ratio: float = 0.3,
                                  decay_factor: float = 0.7, min_length: int = 100,
                                  preserve_original: bool = True) -> str:
    """
    Improved iterative refinement with dynamic ratio adjustment and original content preservation.

    Args:
        text: Original text to summarize
        iterations: Maximum number of iterations
        initial_ratio: Initial extraction ratio
        decay_factor: How much to reduce the extraction ratio each iteration
        min_length: Minimum length (in chars) before stopping iterations
        preserve_original: Whether to include original text in the refinement

    Returns:
        Refined summary
    """
    current = text
    current_ratio = initial_ratio

    # Store original key sentences for later use if preserve_original is True
    if preserve_original:
        original_key_sentences = extract_textrank(text, ratio=0.15)

    for i in range(iterations):
        # Stop if text becomes too short
        if len(current) < min_length:
            print(
                f"Stopping at iteration {i} - text too short: {len(current)} chars")
            break

        # Extract with decreasing ratio
        extracted = extract_textrank(current, ratio=current_ratio)

        # For iterations after first, consider adding some original content
        if preserve_original and i > 0:
            extracted = extracted + " " + original_key_sentences

        # Generate abstract summary
        current = generate_bart(extracted,
                                min_len=min(40, max(20, len(extracted) // 4)),
                                max_len=min(150, max(50, len(extracted) // 2)))

        # Reduce extraction ratio for next iteration
        current_ratio *= decay_factor

    return current


def iterative_refinement_with_early_stopping(text: str, max_iterations: int = 3) -> str:
    """
    Iterative refinement with quality monitoring and early stopping.

    Args:
        text: Original text
        max_iterations: Maximum number of iterations

    Returns:
        Best summary found
    """
    # Get adaptive parameters
    params = adaptive_parameters(text)

    current = text
    best_summary = text
    best_score = 0

    # Initialize with adaptive parameters
    current_ratio = params['initial_ratio']

    # Store original key sentences
    if params['preserve_original']:
        original_key_sentences = extract_textrank(text, ratio=0.15)

    # Track summaries and scores
    summaries = [text]
    scores = []

    for i in range(max_iterations):
        # Stop if text becomes too short
        if len(current) < params['min_length']:
            print(
                f"Stopping at iteration {i} - text too short: {len(current)} chars")
            break

        # Extract key sentences
        extracted = extract_textrank(current, ratio=current_ratio)

        # Consider adding original content
        if params['preserve_original'] and i > 0:
            extracted = extracted + " " + original_key_sentences

        # Generate abstract summary
        current = generate_bart(extracted,
                                min_len=min(40, max(20, len(extracted) // 4)),
                                max_len=min(150, max(50, len(extracted) // 2)))

        # Evaluate current summary
        rouge_scores = evaluate_rouge(current, text)
        bert_s = evaluate_bertscore(current, text)

        # Combined score (weighted average)
        combined_score = (rouge_scores['rouge-1'] * 0.2 +
                          rouge_scores['rouge-2'] * 0.3 +
                          rouge_scores['rouge-l'] * 0.2 +
                          bert_s * 0.3)

        scores.append(combined_score)
        summaries.append(current)

        print(f"Iteration {i+1} score: {combined_score:.4f}")

        # Update best if improved
        if combined_score > best_score:
            best_score = combined_score
            best_summary = current
        # Early stopping if score decreases significantly
        elif i > 0 and combined_score < scores[-2] * 0.95:
            print(f"Early stopping at iteration {i+1} - score decreased")
            break

        # Reduce extraction ratio for next iteration
        current_ratio *= params['decay_factor']

    return best_summary


def create_comparison_visualization(summary_df):
    """
    Create and save visualization of summarization methods comparison,
    focusing only on ROUGE and BERTScore metrics.

    Args:
        summary_df: DataFrame with metrics by method
    """
    metrics = ['rouge-1', 'rouge-2', 'rouge-l', 'bertscore']

    # Create figure with bar charts for metrics
    fig = plt.figure(figsize=(14, 10))

    # Bar charts for each metric
    for i, metric in enumerate(metrics, 1):
        ax = fig.add_subplot(2, 2, i)
        bars = ax.bar(summary_df['method'], summary_df[metric], width=0.6)
        ax.set_title(f"{metric.upper()} by Method")
        ax.set_ylim(0, summary_df[metric].max() * 1.2)
        ax.set_xticklabels(summary_df['method'], rotation=25, ha='right')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Score')

    plt.tight_layout()
    plt.suptitle('Comparison of Summarization Methods', y=1.02, fontsize=16)

    # Save figure
    fig.savefig('textrank_bart_iterative_refinement.png',
                format='png', dpi=300, bbox_inches='tight')
    plt.show()


def enhanced_summarize_feed(rss_url: str, num_articles: int = 1):
    """
    Process RSS feed with enhanced summarization techniques.
    """
    entries = fetch_rss_entries(rss_url, max_articles=num_articles)
    records = []

    # For storing sample summaries
    sample_summaries = []

    for i, entry in enumerate(entries):
        raw = entry.get('content', [{}])[0].get(
            'value') or entry.get('summary', '')
        content = clean_html(raw)
        title = entry.get('title', f"Article {i+1}")

        if len(content.split()) < 50:
            continue

        print(f"\nProcessing: {title}")
        print(f"Word count: {len(content.split())}")

        # Original methods
        ext = extract_textrank(content, ratio=0.2)
        abs_ = generate_bart(content)
        original_it_ref = original_iterative_refinement(
            content, iterations=2, extract_ratio=0.2)

        # New methods
        diverse_ext = extract_diverse_sentences(
            content, ratio=0.25, diversity_penalty=0.6)
        adaptive_it_ref = iterative_refinement_with_early_stopping(
            content, max_iterations=3)

        # Evaluate all methods
        methods = {
            'Extractive (TextRank)': ext,
            'Abstractive (BART)': abs_,
            'Original Iterative': original_it_ref,
            'Diverse Extractive': diverse_ext,
            'Enhanced Iterative': adaptive_it_ref
        }

        # Store sample summaries from first article
        if i == 0:
            sample_summaries = [(name, summary)
                                for name, summary in methods.items()]

        # Evaluate methods
        article_results = evaluate_all_metrics(methods, content)
        records.extend(article_results)

    # Create DataFrame and calculate averages
    df = pd.DataFrame(records)
    summary_df = df.groupby('method').mean().reset_index()

    # Add average length
    summary_df['avg_length'] = df.groupby('method')['length'].mean().values

    print("\n=== Average Metrics by Method ===")
    print(summary_df.to_string(index=False))

    # Create visualization
    create_comparison_visualization(summary_df)

    # Print example summaries
    print("\n=== Example Summaries (First Article) ===")
    for name, summary in sample_summaries:
        word_count = len(summary.split())
        print(f"\n--- {name} ({word_count} words) ---")
        print(summary)

    return summary_df


def main():
    """Main function."""
    print("Enhanced Text Summarization System")
    print("=================================")
    print(f"Using device: {device}")

    rss_url = input("Enter RSS feed URL or local file path: ").strip()
    start = time.time()

    try:
        enhanced_summarize_feed(rss_url, num_articles=5)
        print(f"\nTotal time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
