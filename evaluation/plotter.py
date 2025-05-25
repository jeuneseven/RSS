"""
plotter.py (improved for Parallel, variable n_methods)
- Adapts number of bars/groups for all modes
- For Parallel 1+1: 3 bars (extractive, abstractive, hybrid)
- For Parallel 3+1: 5 bars (3 extractives, abstractive, hybrid)
- For Parallel 3+3: 3 bars (best extractive, best abstractive, hybrid)
- Makes figure width proportional to n_methods (so all bar labels are readable)
- Always uses same logic for main.py method_names/score_dicts order
"""

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, font_size=12):
        self.font_size = font_size

    def plot_multi_panel(self, method_names, score_dicts, output_png='results.png', title='Summarization Comparison'):
        """
        Draws four subplots for ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1.
        :param method_names: list of str (algorithm/model names)
        :param score_dicts: list of dict (average scores for each method)
        :param output_png: output image filename
        :param title: plot super-title
        """
        metrics = [
            ('rouge_rouge-1_f', 'ROUGE-1 F1'),
            ('rouge_rouge-2_f', 'ROUGE-2 F1'),
            ('rouge_rouge-l_f', 'ROUGE-L F1'),
            ('bertscore_f1', 'BERTScore F1')
        ]
        n_methods = len(method_names)
        x = np.arange(n_methods)
        bar_width = 0.65 if n_methods <= 4 else 0.5
        fig_width = max(2.5 * n_methods, 10)
        fig, axs = plt.subplots(2, 2, figsize=(fig_width, 8))
        fig.suptitle(title, fontsize=self.font_size+6)
        for i, (score_key, ylabel) in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axs[row, col]
            values = [score_dict.get(score_key, 0.0)
                      for score_dict in score_dicts]
            bars = ax.bar(x, values, width=bar_width, tick_label=method_names)
            ax.set_title(ylabel, fontsize=self.font_size+2)
            ax.set_ylabel('Score', fontsize=self.font_size)
            ax.set_ylim(0, 1.05)
            for idx, v in enumerate(values):
                ax.text(idx, v + 0.01, f'{v:.3f}',
                        ha='center', fontsize=self.font_size-2)
            ax.set_xticks(x)
            ax.set_xticklabels(
                method_names, fontsize=self.font_size+1, rotation=20 if n_methods > 4 else 0)
            ax.tick_params(axis='y', labelsize=self.font_size)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"Plot saved to {output_png}")
