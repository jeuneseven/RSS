import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import torch
import argparse


def evaluate_summary(reference_text, summary_text):
    """
    Evaluate the quality of a summary compared to a reference text

    Args:
        reference_text (str): The original/reference text
        summary_text (str): The generated summary text

    Returns:
        dict: Dictionary containing evaluation metrics (F1 scores only)
    """
    results = {}

    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, summary_text)

    # Add only the F1 scores to results
    for metric, scores in rouge_scores.items():
        results[f"{metric}_f1"] = scores.fmeasure

    # Calculate BERTScore
    print("Calculating BERTScore...")
    # Ensure CUDA is used if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, F1 = score([summary_text], [reference_text],
                     lang="zh", device=device)

    # Add only the F1 score for BERTScore
    results["bertscore_f1"] = F1.item()

    return results


def format_results(results):
    """Format results as a readable string - showing only final scores"""
    output = "Evaluation Results:\n" + "-" * 50 + "\n"

    # ROUGE scores (only F1)
    output += "ROUGE Scores:\n"
    output += f"  - ROUGE-1: {results['rouge1_f1']:.4f}\n"
    output += f"  - ROUGE-2: {results['rouge2_f1']:.4f}\n"
    output += f"  - ROUGE-L: {results['rougeL_f1']:.4f}\n"

    # BERTScore (only F1)
    output += f"  - BERTScore: {results['bertscore_f1']:.4f}\n"

    return output


def main():
    parser = argparse.ArgumentParser(description='Evaluate Summary Quality')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference text file')
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to summary text file')
    args = parser.parse_args()

    # Example texts (if files not provided)
    default_reference = """
On Tuesday, a publication called Nexus Point News published a headline for what looked to be a major piece of entertainment news: "EXCLUSIVE: Alex Garland Set To Direct 'Elden Ring' Film For A24." The article claimed that Elden Ring "is expected to be Garland's next film" and that shooting is expected to start in 2026.

But don't get your hopes up: soon after the article was published, Nexus Point News pulled it, and the publication won't say why. How this article made the rounds is an interesting case study in the way news travels on the internet.

We first saw the Nexus Point News article from a Wario64 post on X. Generally, Wario64 is a great account to follow to see breaking news in video games almost as soon as it happens. And the account is quite popular, with more than 1 million followers on X. But none of us at The Verge had ever heard of Nexus Point News, let alone recognized it as a place that reliably breaks entertainment news, so something didn't seem right.

(Another recent Nexus Point News "exclusive" includes a potential cast member "being eyed" for one of the male leads of Star Wars: Starfighter, but the article includes an update that says "a LucasFilm spokesperso
    """

    default_summary = """
A Nexus Point News article claiming director Alex Garland would helm an "Elden Ring" film for A24 was published and quickly removed. The story spread when popular gaming news account Wario64 (1M+ followers) shared it on X. The Verge found the source suspicious, noting Nexus Point News lacks credibility as an entertainment news breaker. This incident demonstrates how unverified information can rapidly circulate online through established channels, even when the original source lacks reliability. The article's retraction without explanation further highlights issues with online news verification.
    """

    # Read file contents (if provided)
    reference_text = default_reference
    summary_text = default_summary

    if args.reference:
        try:
            with open(args.reference, 'r', encoding='utf-8') as f:
                reference_text = f.read()
        except Exception as e:
            print(f"Error reading reference text file: {e}")
            return

    if args.summary:
        try:
            with open(args.summary, 'r', encoding='utf-8') as f:
                summary_text = f.read()
        except Exception as e:
            print(f"Error reading summary text file: {e}")
            return

    # Display the texts being used
    print("\nOriginal Text:")
    print("-" * 50)
    print(reference_text)

    print("\nSummary:")
    print("-" * 50)
    print(summary_text)

    # Evaluate the summary
    results = evaluate_summary(reference_text, summary_text)

    # Display results
    print("\n" + format_results(results))


if __name__ == "__main__":
    main()
