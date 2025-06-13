"""
LLM Comparison Bar Chart Generator
Creates a multi-panel bar chart comparing ChatGPT 4o, Gemini 2.5, and Claude 4.0
using the same style as the project's evaluation plotter.

Usage: python llm_comparison_plot.py
"""

import matplotlib.pyplot as plt
import numpy as np


def create_llm_comparison_chart(output_png='llm_comparison.png'):
    """
    Create a 4-panel bar chart comparing LLM performance on different metrics.
    Uses the same style as the project's evaluation plotter.
    """

    # LLM names and their corresponding scores
    method_names = ['ChatGPT 4o', 'Gemini 2.5', 'Claude 4.0']

    # Score data from the table
    rouge_1_scores = [0.1718, 0.1895, 0.2333]
    rouge_2_scores = [0.0906, 0.0845, 0.1106]
    rouge_l_scores = [0.1023, 0.1248, 0.1344]
    bert_scores = [0.8478, 0.8429, 0.8530]

    # Metrics configuration (metric_key, display_name)
    metrics = [
        (rouge_1_scores, 'ROUGE-1 F1'),
        (rouge_2_scores, 'ROUGE-2 F1'),
        (rouge_l_scores, 'ROUGE-L F1'),
        (bert_scores, 'BERTScore F1')
    ]

    # Chart configuration
    n_methods = len(method_names)
    x = np.arange(n_methods)
    bar_width = 0.65
    font_size = 12

    # Create figure with 2x2 subplots
    fig_width = max(2.5 * n_methods, 10)
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, 8))
    fig.suptitle('LLMs Summarization Performance Comparison',
                 fontsize=font_size+6)

    # Create each subplot
    for i, (values, ylabel) in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axs[row, col]

        # Create bars with different colors for each LLM
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        bars = ax.bar(x, values, width=bar_width, color=colors)

        # Configure subplot
        ax.set_title(ylabel, fontsize=font_size+2)
        ax.set_ylabel('Score', fontsize=font_size)
        ax.set_ylim(0, 1.05)

        # Add value labels on top of bars
        for idx, v in enumerate(values):
            ax.text(idx, v + 0.01, f'{v:.4f}',
                    ha='center', fontsize=font_size-2)

        # Set x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=font_size+1, rotation=0)
        ax.tick_params(axis='y', labelsize=font_size)

        # Style the spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"LLM comparison plot saved to {output_png}")


# Main execution
if __name__ == "__main__":
    # Create standard 4-panel comparison
    create_llm_comparison_chart('llm_comparison.png')

    print("All LLM comparison charts generated successfully!")
