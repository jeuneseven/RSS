import argparse
import feedparser
import requests
import os
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from bert_score import score as bert_score

nltk.download('punkt', quiet=True)


def estimate_token_count(text):
    """
    Rough estimation of token count by splitting on whitespace.
    Returns approximate token count.
    """
    return len(text.split())


def calculate_adaptive_sentence_count(text, target_tokens, reduction_ratio=0.8):
    """
    Calculate sentence count based on target token count.
    Ensures the result stays within MIN_TOKENS and MAX_TOKENS bounds.

    Args:
        text: Input text to analyze
        target_tokens: Target number of tokens for the summary
        reduction_ratio: Ratio for reducing from previous stage (default 0.8)

    Returns:
        Optimal sentence count for summarization
    """
    MIN_TOKENS = 50
    MAX_TOKENS = 150

    sentences = split_sentences(text)
    if not sentences:
        return 1

    # Estimate average tokens per sentence
    total_tokens = estimate_token_count(text)
    avg_tokens_per_sentence = total_tokens / \
        len(sentences) if len(sentences) > 0 else 50

    # Calculate target sentence count
    target_sentence_count = max(
        1, int(target_tokens / avg_tokens_per_sentence))

    # Ensure we don't exceed available sentences
    target_sentence_count = min(target_sentence_count, len(sentences))

    # Verify token bounds by estimating result length
    estimated_tokens = target_sentence_count * avg_tokens_per_sentence

    # Adjust if below minimum
    if estimated_tokens < MIN_TOKENS and target_sentence_count < len(sentences):
        target_sentence_count = min(len(sentences), int(
            MIN_TOKENS / avg_tokens_per_sentence) + 1)

    # Adjust if above maximum
    if estimated_tokens > MAX_TOKENS:
        target_sentence_count = max(
            1, int(MAX_TOKENS / avg_tokens_per_sentence))

    return max(1, target_sentence_count)


def fetch_rss_articles(rss_path_or_url, max_articles=5):
    """
    Parse RSS from file or url, return list of (title, content).
    """
    if rss_path_or_url.startswith("http"):
        feed = feedparser.parse(rss_path_or_url)
    else:
        with open(rss_path_or_url, 'r', encoding='utf-8') as f:
            feed = feedparser.parse(f.read())
    articles = []
    for entry in feed.entries[:max_articles]:
        title = entry.get('title', '')
        # Try multiple fields for content
        content = entry.get('content', [{'value': ''}])[0].get(
            'value', '') or entry.get('summary', '')
        # Remove HTML tags
        content = re.sub('<[^<]+?>', '', content)
        articles.append((title, content))
    return articles


def split_sentences(text):
    """
    Split text into sentences.
    """
    return nltk.sent_tokenize(text)


def textrank_summary(text, target_tokens=None, sentence_count=None):
    """
    Apply TextRank summarizer with adaptive sentence count.
    Prioritizes target_tokens over sentence_count if both provided.
    """
    if target_tokens:
        sentence_count = calculate_adaptive_sentence_count(text, target_tokens)
    elif sentence_count is None:
        sentence_count = 10  # Default fallback

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])


def lexrank_summary(text, target_tokens=None, sentence_count=None):
    """
    Apply LexRank summarizer with adaptive sentence count.
    Prioritizes target_tokens over sentence_count if both provided.
    """
    if target_tokens:
        sentence_count = calculate_adaptive_sentence_count(text, target_tokens)
    elif sentence_count is None:
        sentence_count = 8  # Default fallback

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])


def lsa_summary(text, target_tokens=None, sentence_count=None):
    """
    Apply LSA summarization with adaptive sentence count.
    Prioritizes target_tokens over sentence_count if both provided.
    """
    if target_tokens:
        sentence_count = calculate_adaptive_sentence_count(text, target_tokens)
    elif sentence_count is None:
        sentence_count = 6  # Default fallback

    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return ""

    k = min(sentence_count, n)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    if X.shape[0] < 2:
        return ' '.join(sentences[:k])

    svd = TruncatedSVD(n_components=1)
    svd.fit(X)
    component = svd.components_[0]
    # Only select indices in bounds
    top_indices = [idx for idx in np.argsort(component)[::-1] if idx < n][:k]
    selected_sentences = [sentences[idx] for idx in sorted(top_indices)]
    return ' '.join(selected_sentences)


def generate_bart_summary(prompt, target_tokens=150):
    """
    Use BART to generate summary with target token length.
    Ensures output stays within MIN_TOKENS and MAX_TOKENS bounds.
    """
    MIN_TOKENS = 50
    MAX_TOKENS = 150

    # Clamp target_tokens to valid range
    max_length = max(MIN_TOKENS, min(target_tokens, MAX_TOKENS))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn')
    inputs = tokenizer([prompt], max_length=1024,
                       truncation=True, return_tensors='pt')
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        min_length=min(MIN_TOKENS, max_length),
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def compute_rouge(reference, summary):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {k: v.fmeasure for k, v in scores.items()}


def compute_bertscore(reference, summary):
    """
    Compute BERTScore F1.
    """
    P, R, F1 = bert_score([summary], [reference],
                          lang="en", rescale_with_baseline=True)
    return F1[0].item()


def iterative_refinement_summarization(article_text, reduction_ratio=0.8):
    """
    4-stage iterative refinement with proportional token reduction.
    Each stage targets 80% of the previous stage's token count.
    Ensures all outputs stay within 50-150 token bounds.

    Args:
        article_text: Original article text
        reduction_ratio: Ratio for token reduction between stages (default 0.8)

    Returns:
        Dictionary containing summaries and scores for each stage
    """
    MIN_TOKENS = 50
    MAX_TOKENS = 150

    results = {}

    # Calculate initial target tokens (start with a reasonable portion of original)
    original_tokens = estimate_token_count(article_text)

    # Stage 1 target: reasonable starting point, but within bounds
    stage1_target = min(MAX_TOKENS, max(
        MIN_TOKENS, int(original_tokens * 0.3)))

    # Stage 1: TextRank
    tr_summary = textrank_summary(article_text, target_tokens=stage1_target)
    tr_tokens = estimate_token_count(tr_summary)
    tr_rouge = compute_rouge(article_text, tr_summary)
    tr_bert = compute_bertscore(article_text, tr_summary)
    results['TextRank'] = {
        'summary': tr_summary,
        'rouge': tr_rouge,
        'bertscore': tr_bert,
        'tokens': tr_tokens
    }

    # Stage 2: LexRank (target 80% of previous stage)
    stage2_target = max(MIN_TOKENS, int(tr_tokens * reduction_ratio))
    lr_summary = lexrank_summary(tr_summary, target_tokens=stage2_target)
    lr_tokens = estimate_token_count(lr_summary)
    lr_rouge = compute_rouge(article_text, lr_summary)
    lr_bert = compute_bertscore(article_text, lr_summary)
    results['LexRank'] = {
        'summary': lr_summary,
        'rouge': lr_rouge,
        'bertscore': lr_bert,
        'tokens': lr_tokens
    }

    # Stage 3: LSA (target 80% of previous stage)
    stage3_target = max(MIN_TOKENS, int(lr_tokens * reduction_ratio))
    lsa_summary_text = lsa_summary(lr_summary, target_tokens=stage3_target)
    lsa_tokens = estimate_token_count(lsa_summary_text)
    lsa_rouge = compute_rouge(article_text, lsa_summary_text)
    lsa_bert = compute_bertscore(article_text, lsa_summary_text)
    results['LSA'] = {
        'summary': lsa_summary_text,
        'rouge': lsa_rouge,
        'bertscore': lsa_bert,
        'tokens': lsa_tokens
    }

    # Final Stage: BART (target 80% of previous stage, clamped to bounds)
    stage4_target = max(MIN_TOKENS, min(
        MAX_TOKENS, int(lsa_tokens * reduction_ratio)))
    bart_summary_text = generate_bart_summary(
        lsa_summary_text, target_tokens=stage4_target)
    bart_tokens = estimate_token_count(bart_summary_text)
    bart_rouge = compute_rouge(article_text, bart_summary_text)
    bart_bert = compute_bertscore(article_text, bart_summary_text)
    results['BART'] = {
        'summary': bart_summary_text,
        'rouge': bart_rouge,
        'bertscore': bart_bert,
        'tokens': bart_tokens
    }

    return results


def plot_scores(scores_list, direct_scores, out_png='iterative_refinement_result.png'):
    """
    Plot bar chart comparing direct methods vs iterative refinement stages.
    Shows ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore across all methods.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    stages = ['TextRank\n(Direct)', 'LexRank\n(Direct)', 'LSA\n(Direct)', 'BART\n(Direct)',
              'TextRank\n(Iter)', 'LexRank\n(Iter)', 'LSA\n(Iter)', 'BART\n(Iter)']
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

    # Calculate average scores for direct methods
    direct_avg = {method: {metric: np.mean(direct_scores[method][metric])
                           for metric in metrics}
                  for method in ['TextRank', 'LexRank', 'LSA', 'BART']}

    # Calculate average scores for iterative methods
    iter_stages = ['TextRank', 'LexRank', 'LSA', 'BART']
    iter_avg = {stage: {metric: np.mean([s[stage][metric] if metric == 'bertscore' else s[stage]['rouge'][metric]
                                         for s in scores_list]) for metric in metrics}
                for stage in iter_stages}

    x = np.arange(len(stages))
    width = 0.18
    plt.figure(figsize=(16, 8))

    for i, metric in enumerate(metrics):
        # Combine direct and iterative scores
        y_values = []
        for method in ['TextRank', 'LexRank', 'LSA', 'BART']:
            y_values.append(direct_avg[method][metric])  # Direct method
        for method in ['TextRank', 'LexRank', 'LSA', 'BART']:
            y_values.append(iter_avg[method][metric])   # Iterative method

        bars = plt.bar(x + i*width, y_values, width, label=metric_names[i])

        # Add value labels on bars
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha='center', va='bottom', fontsize=8
            )

    plt.xticks(x + 1.5*width, stages, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Direct Methods vs Iterative Refinement: ROUGE/BERTScore Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Iterative Refinement Summarization for RSS with Proportional Token Reduction')
    parser.add_argument('--rss', type=str, required=True,
                        help='RSS URL or local path')
    parser.add_argument('--articles', type=int, default=5,
                        help='Number of articles to process')
    parser.add_argument('--reduction_ratio', type=float, default=0.8,
                        help='Token reduction ratio between stages (default: 0.8)')
    args = parser.parse_args()

    articles = fetch_rss_articles(args.rss, args.articles)
    print(f'Loaded {len(articles)} articles.')

    # Initialize score accumulation for direct methods
    sum_scores = {
        'TextRank': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'LexRank': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'LSA': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'BART': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []}
    }

    all_results = []

    for i, (title, content) in enumerate(articles):
        print(f'\n--- Article {i+1}: {title} ---')
        original_tokens = estimate_token_count(content)
        print(f"Original article tokens: {original_tokens}")
        print(f"\n[Original Article Content]\n{content[:500]}...\n")

        # --- Direct Single-Method Summarization and Evaluation ---
        print(f"\n--- Direct Single-Method Summarization & Evaluation ---")

        # Calculate adaptive targets for direct methods (reasonable fixed targets)
        direct_target = min(150, max(50, int(original_tokens * 0.2)))

        # TextRank Direct
        tr_direct = textrank_summary(content, target_tokens=direct_target)
        tr_direct_tokens = estimate_token_count(tr_direct)
        tr_rouge = compute_rouge(content, tr_direct)
        tr_bert = compute_bertscore(content, tr_direct)
        print(
            f"\n[TextRank Direct] ({tr_direct_tokens} tokens)\nSummary: {tr_direct[:250]}...")
        print(f"ROUGE: {tr_rouge}")
        print(f"BERTScore: {tr_bert:.4f}")
        sum_scores['TextRank']['rouge1'].append(tr_rouge['rouge1'])
        sum_scores['TextRank']['rouge2'].append(tr_rouge['rouge2'])
        sum_scores['TextRank']['rougeL'].append(tr_rouge['rougeL'])
        sum_scores['TextRank']['bertscore'].append(tr_bert)

        # LexRank Direct
        lr_direct = lexrank_summary(content, target_tokens=direct_target)
        lr_direct_tokens = estimate_token_count(lr_direct)
        lr_rouge = compute_rouge(content, lr_direct)
        lr_bert = compute_bertscore(content, lr_direct)
        print(
            f"\n[LexRank Direct] ({lr_direct_tokens} tokens)\nSummary: {lr_direct[:250]}...")
        print(f"ROUGE: {lr_rouge}")
        print(f"BERTScore: {lr_bert:.4f}")
        sum_scores['LexRank']['rouge1'].append(lr_rouge['rouge1'])
        sum_scores['LexRank']['rouge2'].append(lr_rouge['rouge2'])
        sum_scores['LexRank']['rougeL'].append(lr_rouge['rougeL'])
        sum_scores['LexRank']['bertscore'].append(lr_bert)

        # LSA Direct
        lsa_direct = lsa_summary(content, target_tokens=direct_target)
        lsa_direct_tokens = estimate_token_count(lsa_direct)
        lsa_rouge = compute_rouge(content, lsa_direct)
        lsa_bert = compute_bertscore(content, lsa_direct)
        print(
            f"\n[LSA Direct] ({lsa_direct_tokens} tokens)\nSummary: {lsa_direct[:250]}...")
        print(f"ROUGE: {lsa_rouge}")
        print(f"BERTScore: {lsa_bert:.4f}")
        sum_scores['LSA']['rouge1'].append(lsa_rouge['rouge1'])
        sum_scores['LSA']['rouge2'].append(lsa_rouge['rouge2'])
        sum_scores['LSA']['rougeL'].append(lsa_rouge['rougeL'])
        sum_scores['LSA']['bertscore'].append(lsa_bert)

        # BART Direct
        bart_direct = generate_bart_summary(
            content, target_tokens=direct_target)
        bart_direct_tokens = estimate_token_count(bart_direct)
        bart_rouge = compute_rouge(content, bart_direct)
        bart_bert = compute_bertscore(content, bart_direct)
        print(
            f"\n[BART Direct] ({bart_direct_tokens} tokens)\nSummary: {bart_direct[:250]}...")
        print(f"ROUGE: {bart_rouge}")
        print(f"BERTScore: {bart_bert:.4f}")
        sum_scores['BART']['rouge1'].append(bart_rouge['rouge1'])
        sum_scores['BART']['rouge2'].append(bart_rouge['rouge2'])
        sum_scores['BART']['rougeL'].append(bart_rouge['rougeL'])
        sum_scores['BART']['bertscore'].append(bart_bert)

        # --- Iterative (Hybrid) Summarization ---
        print(
            f"\n--- Iterative Refinement Summarization (Ratio: {args.reduction_ratio}) ---")
        results = iterative_refinement_summarization(
            content, args.reduction_ratio)

        for stage in ['TextRank', 'LexRank', 'LSA', 'BART']:
            tokens = results[stage]['tokens']
            print(
                f"\n[{stage} Iterative] ({tokens} tokens)\nSummary: {results[stage]['summary'][:250]}...")
            print(f"ROUGE: {results[stage]['rouge']}")
            print(f"BERTScore: {results[stage]['bertscore']:.4f}")

        all_results.append(results)

    # --- Generate Comparison Plot ---
    plot_scores(all_results, sum_scores,
                out_png='iterative_refinement_result.png')
    print("\nComparison plot saved as iterative_refinement_result.png")

    # --- Output Average Scores Summary ---
    print("\n====== Average Scores across all articles (Direct Methods) ======")
    for method in ['TextRank', 'LexRank', 'LSA', 'BART']:
        r1 = np.mean(sum_scores[method]['rouge1']
                     ) if sum_scores[method]['rouge1'] else 0
        r2 = np.mean(sum_scores[method]['rouge2']
                     ) if sum_scores[method]['rouge2'] else 0
        rl = np.mean(sum_scores[method]['rougeL']
                     ) if sum_scores[method]['rougeL'] else 0
        bs = np.mean(sum_scores[method]['bertscore']
                     ) if sum_scores[method]['bertscore'] else 0
        print(
            f"{method}: ROUGE-1={r1:.4f}, ROUGE-2={r2:.4f}, ROUGE-L={rl:.4f}, BERTScore={bs:.4f}")

    print("\n====== Average Scores across all articles (Iterative Methods) ======")
    iter_stages = ['TextRank', 'LexRank', 'LSA', 'BART']
    for stage in iter_stages:
        r1 = np.mean([r[stage]['rouge']['rouge1'] for r in all_results])
        r2 = np.mean([r[stage]['rouge']['rouge2'] for r in all_results])
        rl = np.mean([r[stage]['rouge']['rougeL'] for r in all_results])
        bs = np.mean([r[stage]['bertscore'] for r in all_results])
        avg_tokens = np.mean([r[stage]['tokens'] for r in all_results])
        print(f"{stage}: ROUGE-1={r1:.4f}, ROUGE-2={r2:.4f}, ROUGE-L={rl:.4f}, BERTScore={bs:.4f} (Avg Tokens: {avg_tokens:.1f})")


if __name__ == '__main__':
    main()
