"""
evaluation/plotter.py

Unified plotting utility for visualizing evaluation results.
- Supports bar plots for multi-method ROUGE/BERTScore comparison.
- Results are saved as PNG files, ready for reporting.
- Accepts average scores or per-article scores as input.
"""

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, font_size=12):
        self.font_size = font_size

    def plot_bar(self, method_names, score_dicts, metrics=None, output_png='results.png', title='Evaluation Comparison'):
        """
        Bar plot for multi-method average metric comparison.
        :param method_names: List of method/algorithm names
        :param score_dicts: List of dicts, each containing metrics (e.g., avg scores)
        :param metrics: List of metrics to plot (default: ROUGE-1 F, ROUGE-2 F, ROUGE-L F, BERTScore F1)
        :param output_png: Output PNG file path
        :param title: Plot title
        """
        # Default metrics
        if metrics is None:
            metrics = [
                ('rouge_rouge-1_f', 'ROUGE-1 F1'),
                ('rouge_rouge-2_f', 'ROUGE-2 F1'),
                ('rouge_rouge-l_f', 'ROUGE-L F1'),
                ('bertscore_f1', 'BERTScore F1')
            ]

        n_methods = len(method_names)
        n_metrics = len(metrics)
        bar_width = 0.15
        x = np.arange(n_methods)

        plt.figure(figsize=(2.5 + n_methods, 6))
        for idx, (score_key, label) in enumerate(metrics):
            values = [score_dict.get(score_key, 0.0)
                      for score_dict in score_dicts]
            plt.bar(x + idx*bar_width, values, width=bar_width, label=label)
            for i, v in enumerate(values):
                plt.text(x[i] + idx*bar_width, v + 0.005,
                         f'{v:.3f}', ha='center', fontsize=self.font_size-2)
        plt.xticks(x + (n_metrics-1)*bar_width/2,
                   method_names, fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.title(title, fontsize=self.font_size+2)
        plt.ylabel('Score', fontsize=self.font_size)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=self.font_size)
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"Plot saved to {output_png}")
