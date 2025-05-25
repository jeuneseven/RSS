"""
basic_diagnostic.py - 基础诊断脚本
专门检查评估系统为什么返回0分
"""

from bert_score import score as bert_score
from rouge import Rouge
from utils.rss_parser import RSSParser
from evaluation.scorer import Evaluator
from models.abstractive import AbstractiveSummarizer
from models.extractive import ExtractiveSummarizer
import sys
import os
sys.path.append('.')


class BasicDiagnostic:
    def __init__(self):
        self.extractive = ExtractiveSummarizer(num_sentences=5)
        self.abstractive = AbstractiveSummarizer()
        self.evaluator = Evaluator()
        self.rouge = Rouge()

    def test_evaluation_system(self):
        """Test if the evaluation system works with simple examples"""
        print("=" * 60)
        print("TESTING EVALUATION SYSTEM")
        print("=" * 60)

        # Test with simple, known examples
        reference = "The cat sat on the mat. The dog ran in the park."
        summary1 = "The cat sat on the mat."  # Should get decent ROUGE score
        summary2 = "A feline was sitting."    # Should get lower ROUGE score
        summary3 = ""                         # Should get 0 score

        test_cases = [
            ("Good overlap", summary1, reference),
            ("Some overlap", summary2, reference),
            ("Empty summary", summary3, reference),
        ]

        for name, summary, ref in test_cases:
            print(f"\nTest: {name}")
            print(f"Reference: {ref}")
            print(f"Summary: '{summary}'")

            # Test direct ROUGE
            try:
                if summary.strip():
                    rouge_scores = self.rouge.get_scores(summary, ref)[0]
                    print(
                        f"Direct ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
                else:
                    print("Direct ROUGE: Skipped (empty summary)")
            except Exception as e:
                print(f"Direct ROUGE Error: {e}")

            # Test direct BERTScore
            try:
                if summary.strip():
                    P, R, F1 = bert_score([summary], [ref], lang='en')
                    print(f"Direct BERTScore F1: {F1.item():.4f}")
                else:
                    print("Direct BERTScore: Skipped (empty summary)")
            except Exception as e:
                print(f"Direct BERTScore Error: {e}")

            # Test through Evaluator class
            try:
                eval_scores = self.evaluator.score(summary, ref)
                print(
                    f"Evaluator ROUGE-1: {eval_scores.get('rouge_rouge-1_f', 'NOT FOUND')}")
                print(
                    f"Evaluator BERTScore: {eval_scores.get('bertscore_f1', 'NOT FOUND')}")
                # Show first 5 keys
                print(f"All Evaluator keys: {list(eval_scores.keys())[:5]}...")
            except Exception as e:
                print(f"Evaluator Error: {e}")

    def test_single_article(self, rss_url):
        """Test with a single real article"""
        print("\n" + "=" * 60)
        print("TESTING WITH REAL ARTICLE")
        print("=" * 60)

        parser = RSSParser()
        articles = parser.parse(rss_url, max_articles=1)

        if not articles:
            print("No articles found!")
            return

        article = articles[0]
        content = article['content']

        print(f"Article title: {article['title']}")
        print(
            f"Content length: {len(content)} characters, {len(content.split())} words")
        print(f"Content preview: {content[:200]}...")

        if len(content.split()) < 30:
            print("Content too short for meaningful testing!")
            return

        # Generate summaries
        print("\nGenerating summaries...")
        try:
            tr_sum = self.extractive.textrank(content)
            print(f"TextRank summary: {tr_sum[:100]}...")
            print(
                f"TextRank length: {len(tr_sum)} chars, {len(tr_sum.split())} words")
        except Exception as e:
            print(f"TextRank generation error: {e}")
            return

        try:
            abs_sum = self.abstractive.bart(content)
            print(f"BART summary: {abs_sum[:100]}...")
            print(
                f"BART length: {len(abs_sum)} chars, {len(abs_sum.split())} words")
        except Exception as e:
            print(f"BART generation error: {e}")
            return

        # Test evaluation with real summaries
        print("\nTesting evaluation with real summaries...")

        # Test TextRank summary
        print(f"\nEvaluating TextRank summary against original content:")
        self.test_single_evaluation(tr_sum, content, "TextRank")

        # Test BART summary
        print(f"\nEvaluating BART summary against original content:")
        self.test_single_evaluation(abs_sum, content, "BART")

        # Test a simple truncated version of original (should score high)
        simple_ref = " ".join(content.split()[:50])
        print(f"\nEvaluating truncated original against full original (should score high):")
        self.test_single_evaluation(simple_ref, content, "Truncated Original")

    def test_single_evaluation(self, summary, reference, name):
        """Test evaluation of a single summary-reference pair"""
        if not summary.strip():
            print(f"{name}: Empty summary, skipping evaluation")
            return

        if not reference.strip():
            print(f"{name}: Empty reference, skipping evaluation")
            return

        print(f"\n{name} Evaluation:")
        print(
            f"  Summary length: {len(summary)} chars, {len(summary.split())} words")
        print(
            f"  Reference length: {len(reference)} chars, {len(reference.split())} words")

        # Direct ROUGE test
        try:
            rouge_scores = self.rouge.get_scores(summary, reference)[0]
            print(f"  Direct ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
            print(f"  Direct ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
            print(f"  Direct ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
        except Exception as e:
            print(f"  Direct ROUGE Error: {e}")

        # Direct BERTScore test
        try:
            P, R, F1 = bert_score([summary], [reference], lang='en')
            print(
                f"  Direct BERTScore - P: {P.item():.4f}, R: {R.item():.4f}, F1: {F1.item():.4f}")
        except Exception as e:
            print(f"  Direct BERTScore Error: {e}")

        # Evaluator class test
        try:
            eval_scores = self.evaluator.score(summary, reference)
            print(f"  Evaluator output keys: {list(eval_scores.keys())}")

            # Try different possible key formats
            possible_rouge_keys = [
                'rouge_rouge-1_f', 'rouge-1_f', 'rouge_1_f',
                'rouge', 'rouge-1', 'rouge_rouge-1'
            ]
            possible_bert_keys = [
                'bertscore_f1', 'bertscore', 'bert_score_f1',
                'bert_score', 'f1', 'bertscore_f'
            ]

            print("  Looking for ROUGE keys:")
            for key in possible_rouge_keys:
                if key in eval_scores:
                    print(f"    Found {key}: {eval_scores[key]}")

            print("  Looking for BERTScore keys:")
            for key in possible_bert_keys:
                if key in eval_scores:
                    print(f"    Found {key}: {eval_scores[key]}")

            # Print all values to see what we actually have
            print("  All evaluator values:")
            for key, value in eval_scores.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {type(value)}")

        except Exception as e:
            print(f"  Evaluator Error: {e}")

    def check_dependencies(self):
        """Check if all required libraries are properly installed"""
        print("=" * 60)
        print("CHECKING DEPENDENCIES")
        print("=" * 60)

        try:
            import rouge
            print("✓ rouge library imported successfully")
        except Exception as e:
            print(f"✗ rouge library error: {e}")

        try:
            import bert_score
            print("✓ bert_score library imported successfully")
        except Exception as e:
            print(f"✗ bert_score library error: {e}")

        try:
            from evaluation.scorer import Evaluator
            evaluator = Evaluator()
            print("✓ Evaluator class imported successfully")
        except Exception as e:
            print(f"✗ Evaluator class error: {e}")


def main():
    diagnostic = BasicDiagnostic()

    print("Basic Evaluation System Diagnostic")
    print("This will help identify why all scores are showing as 0.0000")

    # Check dependencies first
    diagnostic.check_dependencies()

    # Test evaluation system with simple examples
    diagnostic.test_evaluation_system()

    # Test with real article
    rss_url = input(
        "\nEnter RSS feed URL or local file path to test with real article: ")
    if rss_url.strip():
        diagnostic.test_single_article(rss_url)

    print("\n" + "=" * 60)
    print("Diagnostic complete. Please share the output to identify the issue.")
    print("=" * 60)


if __name__ == "__main__":
    main()
