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
    inputs = bart_tokenizer(text, truncation=True,
                            max_length=1024, return_tensors="pt").to(device)
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


def iterative_refinement(text: str, iterations: int = 2, extract_ratio: float = 0.2) -> str:
    """
    Iteratively refine summary: extract then generate.
    """
    current = text
    for _ in range(iterations):
        extracted = extract_textrank(current, ratio=extract_ratio)
        current = generate_bart(extracted)
    return current


def evaluate_rouge(summary: str, reference: str) -> dict:
    """Compute ROUGE scores."""
    if not summary or not reference:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    scores = rouge_evaluator.get_scores(summary, reference)[0]
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }


def evaluate_bertscore(summary: str, reference: str) -> float:
    """Compute BERTScore F1."""
    if not summary or not reference:
        return 0.0
    P, R, F1 = bert_score([summary], [reference], lang='en')
    return F1.item()


def summarize_feed(rss_url: str, num_articles: int = 5):
    """Process RSS feed: generate and evaluate summaries."""
    entries = fetch_rss_entries(rss_url, max_articles=num_articles)
    records = []

    for entry in entries:
        raw = entry.get('content', [{}])[0].get(
            'value') or entry.get('summary', '')
        content = clean_html(raw)
        if len(content.split()) < 50:
            continue
        ext = extract_textrank(content, ratio=0.2)
        abs_ = generate_bart(content)
        it_ref = iterative_refinement(content, iterations=2, extract_ratio=0.2)
        rouge_ext = evaluate_rouge(ext, content)
        rouge_abs = evaluate_rouge(abs_, content)
        rouge_itr = evaluate_rouge(it_ref, content)
        bert_ext = evaluate_bertscore(ext, content)
        bert_abs = evaluate_bertscore(abs_, content)
        bert_itr = evaluate_bertscore(it_ref, content)
        records.extend([
            {'method': 'Extractive (TextRank)', 'rouge-1': rouge_ext['rouge-1'],
             'rouge-2': rouge_ext['rouge-2'], 'rouge-l': rouge_ext['rouge-l'], 'bertscore': bert_ext},
            {'method': 'Abstractive (BART)', 'rouge-1': rouge_abs['rouge-1'],
             'rouge-2': rouge_abs['rouge-2'], 'rouge-l': rouge_abs['rouge-l'], 'bertscore': bert_abs},
            {'method': 'Iterative (TextRank→BART)', 'rouge-1': rouge_itr['rouge-1'],
             'rouge-2': rouge_itr['rouge-2'], 'rouge-l': rouge_itr['rouge-l'], 'bertscore': bert_itr}
        ])

    df = pd.DataFrame(records)
    summary_df = df.groupby('method').mean().reset_index()

    print("\n=== Average Metrics by Method ===")
    print(summary_df.to_string(index=False))

    # Plot and save comparison chart as PNG
    metrics = ['rouge-1', 'rouge-2', 'rouge-l', 'bertscore']
    fig = plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics, 1):
        ax = fig.add_subplot(2, 2, i)
        bars = ax.bar(summary_df['method'], summary_df[metric], width=0.6)
        ax.set_title(f"{metric.upper()} by Method")
        ax.set_ylim(0, summary_df[metric].max() * 1.2)
        ax.set_xticklabels(summary_df['method'], rotation=20)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h +
                    0.01, f'{h:.4f}', ha='center', fontsize=9)
        ax.set_ylabel('Score')

    plt.tight_layout()
    plt.suptitle('Comparison of Summarization Methods (RSS Feed)', y=1.02)
    # Save the figure as a PNG file
    fig.savefig('textrank_bart_iterative_refinement.png',
                format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return summary_df


if __name__ == "__main__":
    rss_url = input("Enter RSS feed URL or local file path: ").strip()
    start = time.time()
    summarize_feed(rss_url, num_articles=5)
    print(f"\nTotal time: {time.time() - start:.2f}s")
