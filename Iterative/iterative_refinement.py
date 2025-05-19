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


def textrank_summary(text, sentence_count=10):
    """
    Apply TextRank summarizer, return concatenated summary.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])


def lexrank_summary(text, sentence_count=8):
    """
    Apply LexRank summarizer, return concatenated summary.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])


def lsa_summary(text, sentence_count=6):
    """
    Apply LSA summarization (SVD on TF-IDF), return concatenated summary.
    Handles case where sentence_count > actual sentences.
    """
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


def generate_bart_summary(prompt, max_length=150):
    """
    Use BART to generate summary from prompt.
    """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn')
    inputs = tokenizer([prompt], max_length=1024,
                       truncation=True, return_tensors='pt')
    summary_ids = model.generate(
        inputs['input_ids'], num_beams=4, max_length=max_length, early_stopping=True)
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


def iterative_refinement_summarization(article_text):
    """
    3-stage iterative refinement: TextRank → LexRank → LSA → BART.
    Returns summaries and scores for each stage.
    """
    results = {}

    # Stage 1: TextRank
    tr_summary = textrank_summary(article_text, sentence_count=10)
    tr_rouge = compute_rouge(article_text, tr_summary)
    tr_bert = compute_bertscore(article_text, tr_summary)
    results['TextRank'] = {'summary': tr_summary,
                           'rouge': tr_rouge, 'bertscore': tr_bert}

    # Stage 2: LexRank (on TextRank summary)
    lr_summary = lexrank_summary(tr_summary, sentence_count=8)
    lr_rouge = compute_rouge(article_text, lr_summary)
    lr_bert = compute_bertscore(article_text, lr_summary)
    results['LexRank'] = {'summary': lr_summary,
                          'rouge': lr_rouge, 'bertscore': lr_bert}

    # Stage 3: LSA (on LexRank summary)
    lsa_summary_text = lsa_summary(lr_summary, sentence_count=6)
    lsa_rouge = compute_rouge(article_text, lsa_summary_text)
    lsa_bert = compute_bertscore(article_text, lsa_summary_text)
    results['LSA'] = {'summary': lsa_summary_text,
                      'rouge': lsa_rouge, 'bertscore': lsa_bert}

    # Final Stage: BART (on LSA summary)
    bart_summary_text = generate_bart_summary(lsa_summary_text)
    bart_rouge = compute_rouge(article_text, bart_summary_text)
    bart_bert = compute_bertscore(article_text, bart_summary_text)
    results['BART'] = {'summary': bart_summary_text,
                       'rouge': bart_rouge, 'bertscore': bart_bert}

    return results


def plot_scores(scores_list, out_png='iterative_refinement_result.png'):
    """
    Plot bar chart of ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore across stages,
    with value labels on each bar.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    stages = ['TextRank', 'LexRank', 'LSA', 'BART']
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

    # Average scores for all articles per stage
    avg_scores = {stage: {metric: np.mean([s[stage][metric] if metric == 'bertscore' else s[stage]['rouge'][metric]
                                           for s in scores_list]) for metric in metrics}
                  for stage in stages}

    x = np.arange(len(stages))
    width = 0.18
    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        y = [avg_scores[stage][metric] for stage in stages]
        bars = plt.bar(x + i*width, y, width, label=metric_names[i])
        # Add value label on top of each bar
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.4f}",
                ha='center', va='bottom', fontsize=10
            )

    plt.xticks(x + 1.5*width, stages)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Iterative Summarization: ROUGE/BERTScore Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Iterative Refinement Summarization for RSS')
    parser.add_argument('--rss', type=str, required=True,
                        help='RSS URL or local path')
    parser.add_argument('--articles', type=int, default=5,
                        help='Number of articles')
    args = parser.parse_args()

    articles = fetch_rss_articles(args.rss, args.articles)
    print(f'Loaded {len(articles)} articles.')

    # For accumulating scores for the four direct methods
    sum_scores = {
        'TextRank': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'LexRank': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'LSA': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []},
        'BART': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []}
    }

    all_results = []
    for i, (title, content) in enumerate(articles):
        print(f'\n--- Article {i+1}: {title} ---')

        # --- Direct Single-Method Summarization and Evaluation ---
        print(f"\n--- Direct Single-Method Summarization & Evaluation ---")

        # TextRank
        tr_direct = textrank_summary(content, sentence_count=10)
        tr_rouge = compute_rouge(content, tr_direct)
        tr_bert = compute_bertscore(content, tr_direct)
        print(f"\n[TextRank]\nSummary: {tr_direct[:250]}...")
        print(f"ROUGE: {tr_rouge}")
        print(f"BERTScore: {tr_bert:.4f}")
        sum_scores['TextRank']['rouge1'].append(tr_rouge['rouge1'])
        sum_scores['TextRank']['rouge2'].append(tr_rouge['rouge2'])
        sum_scores['TextRank']['rougeL'].append(tr_rouge['rougeL'])
        sum_scores['TextRank']['bertscore'].append(tr_bert)

        # LexRank
        lr_direct = lexrank_summary(content, sentence_count=8)
        lr_rouge = compute_rouge(content, lr_direct)
        lr_bert = compute_bertscore(content, lr_direct)
        print(f"\n[LexRank]\nSummary: {lr_direct[:250]}...")
        print(f"ROUGE: {lr_rouge}")
        print(f"BERTScore: {lr_bert:.4f}")
        sum_scores['LexRank']['rouge1'].append(lr_rouge['rouge1'])
        sum_scores['LexRank']['rouge2'].append(lr_rouge['rouge2'])
        sum_scores['LexRank']['rougeL'].append(lr_rouge['rougeL'])
        sum_scores['LexRank']['bertscore'].append(lr_bert)

        # LSA
        lsa_direct = lsa_summary(content, sentence_count=6)
        lsa_rouge = compute_rouge(content, lsa_direct)
        lsa_bert = compute_bertscore(content, lsa_direct)
        print(f"\n[LSA]\nSummary: {lsa_direct[:250]}...")
        print(f"ROUGE: {lsa_rouge}")
        print(f"BERTScore: {lsa_bert:.4f}")
        sum_scores['LSA']['rouge1'].append(lsa_rouge['rouge1'])
        sum_scores['LSA']['rouge2'].append(lsa_rouge['rouge2'])
        sum_scores['LSA']['rougeL'].append(lsa_rouge['rougeL'])
        sum_scores['LSA']['bertscore'].append(lsa_bert)

        # BART
        bart_direct = generate_bart_summary(content)
        bart_rouge = compute_rouge(content, bart_direct)
        bart_bert = compute_bertscore(content, bart_direct)
        print(f"\n[BART]\nSummary: {bart_direct[:250]}...")
        print(f"ROUGE: {bart_rouge}")
        print(f"BERTScore: {bart_bert:.4f}")
        sum_scores['BART']['rouge1'].append(bart_rouge['rouge1'])
        sum_scores['BART']['rouge2'].append(bart_rouge['rouge2'])
        sum_scores['BART']['rougeL'].append(bart_rouge['rougeL'])
        sum_scores['BART']['bertscore'].append(bart_bert)

        # --- Iterative (Hybrid) Summarization ---
        results = iterative_refinement_summarization(content)
        for stage in ['TextRank', 'LexRank', 'LSA', 'BART']:
            print(f"\n[{stage}]\nSummary: {results[stage]['summary'][:250]}...")
            print(f"ROUGE: {results[stage]['rouge']}")
            print(f"BERTScore: {results[stage]['bertscore']:.4f}")
        all_results.append(results)

    # --- Plot and Save Hybrid (Iterative) Method Results ---
    plot_scores(all_results, out_png='iterative_refinement_result.png')
    print("\nComparison plot saved as iterative_refinement_result.png")

    # --- Output Average Scores for Direct Methods ---
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


if __name__ == '__main__':
    main()
