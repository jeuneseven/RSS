"""
llm_comparison_tester.py - Command Line LLM Summary Comparison Tool

Simple file-based tool for comparing LLM summarization capabilities:
1. Save your original text as a file (e.g., original.txt)
2. Save each LLM's summary as separate files (e.g., gpt4.txt, claude.txt, gemini.txt)
3. Run comparison command
4. Get ranked results

Examples:
    # Compare multiple LLMs on the same original text
    python cli_llm_tester.py --original original.txt --summaries gpt4.txt claude.txt gemini.txt

    # Compare with custom names
    python cli_llm_tester.py --original original.txt --summaries gpt4.txt:GPT-4 claude.txt:Claude-3 gemini.txt:Gemini-Pro

    # Single summary evaluation
    python cli_llm_tester.py --original original.txt --summary gpt4.txt

    # Batch mode with directory
    python cli_llm_tester.py --original original.txt --summary-dir summaries/
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path so we can import from evaluation
sys.path.append('.')

try:
    from evaluation.scorer import Evaluator
    USE_PROJECT_EVALUATOR = True
    print("✓ Using project's evaluation system")
except ImportError:
    print("⚠ Project evaluator not found, using standalone version")
    USE_PROJECT_EVALUATOR = False
    # Fallback imports for standalone version
    from rouge import Rouge
    from bert_score import score as bert_score


class CLILLMTester:
    def __init__(self):
        """Initialize the CLI LLM tester"""
        if USE_PROJECT_EVALUATOR:
            self.evaluator = Evaluator()
        else:
            self.rouge = Rouge()

    def load_text_file(self, filepath):
        """Load text from file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                print(f"⚠ Warning: File {filepath} is empty")
                return None
            return content
        except Exception as e:
            print(f"❌ Error reading file {filepath}: {e}")
            return None

    def evaluate_summary(self, original_text, summary_text):
        """Evaluate a summary against the original text"""
        if not original_text.strip() or not summary_text.strip():
            return {
                'rouge_1_f1': 0.0,
                'rouge_2_f1': 0.0,
                'rouge_l_f1': 0.0,
                'bertscore_f1': 0.0,
                'average': 0.0
            }

        if USE_PROJECT_EVALUATOR:
            return self._evaluate_with_project_system(original_text, summary_text)
        else:
            return self._evaluate_with_standalone(original_text, summary_text)

    def _evaluate_with_project_system(self, original_text, summary_text):
        """Use the project's evaluation system"""
        try:
            scores = self.evaluator.score(summary_text, original_text)

            rouge_scores = scores.get('rouge', {})
            bertscore_scores = scores.get('bertscore', {})

            rouge_1 = rouge_scores.get('rouge-1', {}).get('f', 0.0)
            rouge_2 = rouge_scores.get('rouge-2', {}).get('f', 0.0)
            rouge_l = rouge_scores.get('rouge-l', {}).get('f', 0.0)
            bertscore = bertscore_scores.get('f1', 0.0)

            average = (rouge_1 + rouge_2 + rouge_l + bertscore) / 4

            return {
                'rouge_1_f1': rouge_1,
                'rouge_2_f1': rouge_2,
                'rouge_l_f1': rouge_l,
                'bertscore_f1': bertscore,
                'average': average
            }
        except Exception as e:
            print(f"❌ Error using project evaluator: {e}")
            return {'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0, 'bertscore_f1': 0.0, 'average': 0.0}

    def _evaluate_with_standalone(self, original_text, summary_text):
        """Use standalone evaluation system as fallback"""
        try:
            rouge_scores = self.rouge.get_scores(
                summary_text, original_text)[0]
            P, R, F1 = bert_score([summary_text], [original_text], lang='en')

            rouge_1 = rouge_scores['rouge-1']['f']
            rouge_2 = rouge_scores['rouge-2']['f']
            rouge_l = rouge_scores['rouge-l']['f']
            bertscore = F1.item()

            average = (rouge_1 + rouge_2 + rouge_l + bertscore) / 4

            return {
                'rouge_1_f1': rouge_1,
                'rouge_2_f1': rouge_2,
                'rouge_l_f1': rouge_l,
                'bertscore_f1': bertscore,
                'average': average
            }
        except Exception as e:
            print(f"❌ Error in standalone evaluation: {e}")
            return {'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0, 'bertscore_f1': 0.0, 'average': 0.0}

    def parse_summary_spec(self, summary_spec):
        """Parse summary specification (filepath or filepath:name)"""
        if ':' in summary_spec:
            filepath, name = summary_spec.rsplit(':', 1)
            return filepath.strip(), name.strip()
        else:
            filepath = summary_spec.strip()
            # Use filename without extension as name
            name = Path(filepath).stem
            return filepath, name

    def compare_summaries(self, original_file, summary_specs, output_file=None):
        """Compare multiple summaries against the original text"""
        # Load original text
        original_text = self.load_text_file(original_file)
        if not original_text:
            print(f"❌ Failed to load original text from {original_file}")
            return False

        original_words = len(original_text.split())
        print(f"📄 Original text loaded: {original_words} words")
        print(f"   File: {original_file}")

        # Load and evaluate summaries
        results = {}
        summaries = {}

        print(f"\n🔄 Evaluating {len(summary_specs)} summaries...")
        print("-" * 60)

        for spec in summary_specs:
            filepath, llm_name = self.parse_summary_spec(spec)

            summary_text = self.load_text_file(filepath)
            if not summary_text:
                print(f"❌ Skipping {llm_name}: Failed to load {filepath}")
                continue

            summary_words = len(summary_text.split())
            print(f"📝 {llm_name}: {summary_words} words ({filepath})")

            # Evaluate summary
            scores = self.evaluate_summary(original_text, summary_text)
            results[llm_name] = scores
            summaries[llm_name] = summary_text

            print(f"   ✅ Score: {scores['average']:.4f}")

        if not results:
            print("❌ No valid summaries to compare!")
            return False

        # Display results
        self.display_comparison_results(
            original_file, original_words, results, summaries)

        # Save results if requested
        if output_file:
            self.save_results_to_file(
                original_file, original_text, results, summaries, output_file)

        return True

    def evaluate_single_summary(self, original_file, summary_file, output_file=None):
        """Evaluate a single summary against the original text"""
        # Load texts
        original_text = self.load_text_file(original_file)
        summary_text = self.load_text_file(summary_file)

        if not original_text or not summary_text:
            print("❌ Failed to load required files")
            return False

        original_words = len(original_text.split())
        summary_words = len(summary_text.split())

        print(f"📄 Original: {original_words} words ({original_file})")
        print(f"📝 Summary:  {summary_words} words ({summary_file})")

        # Evaluate
        print(f"\n🔄 Evaluating summary...")
        scores = self.evaluate_summary(original_text, summary_text)

        # Display results
        compression_ratio = (
            original_words - summary_words) / original_words * 100

        print(f"\n" + "="*70)
        print(f"📊 EVALUATION RESULTS")
        print(f"="*70)
        print(f"Summary file:      {summary_file}")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"-"*70)
        print(f"ROUGE-1 F1:       {scores['rouge_1_f1']:.4f}")
        print(f"ROUGE-2 F1:       {scores['rouge_2_f1']:.4f}")
        print(f"ROUGE-L F1:       {scores['rouge_l_f1']:.4f}")
        print(f"BERTScore F1:     {scores['bertscore_f1']:.4f}")
        print(f"-"*70)
        print(f"AVERAGE SCORE:    {scores['average']:.4f}")
        print(f"="*70)

        # Save results if requested
        if output_file:
            self.save_single_result_to_file(
                original_file, summary_file, original_text, summary_text, scores, output_file)

        return True

    def batch_compare_directory(self, original_file, summary_dir, output_file=None):
        """Compare all summary files in a directory"""
        summary_dir_path = Path(summary_dir)
        if not summary_dir_path.exists() or not summary_dir_path.is_dir():
            print(
                f"❌ Directory {summary_dir} does not exist or is not a directory")
            return False

        # Find all text files in the directory
        text_files = list(summary_dir_path.glob("*.txt"))
        if not text_files:
            print(f"❌ No .txt files found in {summary_dir}")
            return False

        print(f"📁 Found {len(text_files)} summary files in {summary_dir}")

        # Create summary specs
        summary_specs = []
        for file_path in sorted(text_files):
            summary_specs.append(str(file_path))

        return self.compare_summaries(original_file, summary_specs, output_file)

    def display_comparison_results(self, original_file, original_words, results, summaries):
        """Display comparison results in a formatted table"""
        print(f"\n" + "="*90)
        print(f"🏆 LLM COMPARISON RESULTS")
        print(f"="*90)
        print(f"Original file: {original_file}")
        print(f"Original length: {original_words} words")
        print(f"LLMs compared: {len(results)}")

        # Results table header
        print(f"\n{'Rank':<4} {'LLM':<20} {'ROUGE-1':<9} {'ROUGE-2':<9} {'ROUGE-L':<9} {'BERTScore':<10} {'Average':<9} {'Words':<6} {'Compress':<8}")
        print("-"*90)

        # Sort by average score
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]['average'], reverse=True)

        for rank, (llm_name, scores) in enumerate(sorted_results, 1):
            summary_words = len(summaries[llm_name].split())
            compression = (original_words - summary_words) / \
                original_words * 100

            # Add medal emojis for top 3
            display_name = llm_name
            if rank == 1:
                display_name += " 🥇"
            elif rank == 2:
                display_name += " 🥈"
            elif rank == 3:
                display_name += " 🥉"

            print(f"{rank:<4} {display_name:<20} {scores['rouge_1_f1']:<9.4f} "
                  f"{scores['rouge_2_f1']:<9.4f} {scores['rouge_l_f1']:<9.4f} "
                  f"{scores['bertscore_f1']:<10.4f} {scores['average']:<9.4f} "
                  f"{summary_words:<6} {compression:<8.1f}%")

        print("-"*90)

        # Winner details
        winner = sorted_results[0]
        print(f"\n🎉 WINNER: {winner[0]}")
        print(f"   🎯 Best Average Score: {winner[1]['average']:.4f}")
        print(f"   📊 Breakdown: ROUGE-1({winner[1]['rouge_1_f1']:.4f}) + "
              f"ROUGE-2({winner[1]['rouge_2_f1']:.4f}) + "
              f"ROUGE-L({winner[1]['rouge_l_f1']:.4f}) + "
              f"BERTScore({winner[1]['bertscore_f1']:.4f})")

        print("="*90)

    def save_results_to_file(self, original_file, original_text, results, summaries, output_file):
        """Save comparison results to JSON file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'original_file': original_file,
            'original_length': len(original_text.split()),
            'num_llms': len(results),
            'results': results,
            'summaries': {name: {'text': text, 'length': len(text.split())}
                          for name, text in summaries.items()}
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def save_single_result_to_file(self, original_file, summary_file, original_text, summary_text, scores, output_file):
        """Save single evaluation result to JSON file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'original_file': original_file,
            'summary_file': summary_file,
            'original_length': len(original_text.split()),
            'summary_length': len(summary_text.split()),
            'scores': scores
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='CLI LLM Summary Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare multiple LLM summaries
    python cli_llm_tester.py --original news.txt --summaries gpt4.txt claude.txt gemini.txt
    
    # Compare with custom names
    python cli_llm_tester.py --original news.txt --summaries gpt4.txt:GPT-4 claude.txt:Claude-3
    
    # Single summary evaluation
    python cli_llm_tester.py --original news.txt --summary gpt4.txt
    
    # Batch compare directory
    python cli_llm_tester.py --original news.txt --summary-dir summaries/
    
    # Save results to file
    python cli_llm_tester.py --original news.txt --summaries *.txt --output results.json
        """
    )

    parser.add_argument('--original', '-o', required=True,
                        help='Path to original text file')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--summary', '-s',
                       help='Path to single summary file for evaluation')
    group.add_argument('--summaries', '-m', nargs='+',
                       help='Paths to multiple summary files for comparison (format: file.txt or file.txt:CustomName)')
    group.add_argument('--summary-dir', '-d',
                       help='Directory containing summary files to compare')

    parser.add_argument('--output', '-out',
                        help='Save results to JSON file')

    args = parser.parse_args()

    tester = CLILLMTester()

    print("🤖 CLI LLM Summary Comparison Tool")
    print("="*50)

    success = False

    if args.summary:
        # Single summary evaluation
        success = tester.evaluate_single_summary(
            args.original, args.summary, args.output)

    elif args.summaries:
        # Multiple summaries comparison
        success = tester.compare_summaries(
            args.original, args.summaries, args.output)

    elif args.summary_dir:
        # Batch directory comparison
        success = tester.batch_compare_directory(
            args.original, args.summary_dir, args.output)

    if success:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
