"""
plotter.py (Enhanced with Classification Visualization)
- Adds classification dashboard functionality
- Maintains compatibility with existing summarization plots
- Provides comprehensive analysis of classification results
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class Plotter:
    def __init__(self, font_size=12):
        self.font_size = font_size
        # Define colors for different categories
        self.category_colors = {
            'technology': '#FF6B6B',
            'politics': '#4ECDC4',
            'business': '#45B7D1',
            'entertainment': '#96CEB4',
            'health': '#FFEAA7',
            'sports': '#DDA0DD',
            'unknown': '#95A5A6'
        }

    def plot_multi_panel(self, method_names, score_dicts, output_png='results.png', title='Summarization Comparison'):
        """
        Draws four subplots for ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1.
        (Original summarization plotting functionality - unchanged)
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

    def plot_classification_dashboard(self, articles_data, output_png='classification_dashboard.png',
                                      title='RSS Article Classification Analysis Dashboard'):
        """
        Creates a comprehensive classification analysis dashboard with four panels.

        Args:
            articles_data: List of article dictionaries containing classification results
            output_png: Output filename for the dashboard
            title: Main title for the dashboard
        """
        # Extract classification data
        categories = []
        confidences = []
        article_titles = []
        classification_methods = []

        for article in articles_data:
            # Get classification data with fallbacks
            category = article.get('category', 'unknown')
            confidence = article.get('classification_confidence', 0.0)
            title_text = article.get('title', 'Unknown Article')
            method = article.get('classification_method', 'unknown')

            categories.append(category)
            confidences.append(float(confidence))
            # Truncate long titles for display
            short_title = title_text[:40] + \
                '...' if len(title_text) > 40 else title_text
            article_titles.append(short_title)
            classification_methods.append(method)

        # Create dashboard with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=self.font_size + 8, y=0.98)

        # Panel 1 (Top-Left): Category Distribution Pie Chart
        category_counts = Counter(categories)
        category_names = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = [self.category_colors.get(
            cat, '#95A5A6') for cat in category_names]

        wedges, texts, autotexts = ax1.pie(counts, labels=category_names, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax1.set_title('Article Distribution by Category',
                      fontsize=self.font_size + 2, pad=20)

        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(self.font_size - 1)
            autotext.set_weight('bold')

        # Panel 2 (Top-Right): Classification Confidence by Article
        # Sort articles by confidence for better visualization
        sorted_data = sorted(
            zip(article_titles, confidences, categories), key=lambda x: x[1])
        sorted_titles, sorted_confidences, sorted_categories = zip(
            *sorted_data)

        y_pos = np.arange(len(sorted_titles))
        bar_colors = [self.category_colors.get(
            cat, '#95A5A6') for cat in sorted_categories]

        bars = ax2.barh(y_pos, sorted_confidences, color=bar_colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_titles, fontsize=self.font_size - 2)
        ax2.set_xlabel('Classification Confidence', fontsize=self.font_size)
        ax2.set_title('Classification Confidence by Article',
                      fontsize=self.font_size + 2, pad=20)
        ax2.set_xlim(0, 1.0)

        # Add confidence value labels on bars
        for i, (bar, conf) in enumerate(zip(bars, sorted_confidences)):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{conf:.3f}', ha='left', va='center', fontsize=self.font_size - 3)

        # Panel 3 (Bottom-Left): Confidence Level Distribution
        confidence_levels = []
        for conf in confidences:
            if conf >= 0.8:
                confidence_levels.append('High (≥0.8)')
            elif conf >= 0.6:
                confidence_levels.append('Medium (0.6-0.8)')
            else:
                confidence_levels.append('Low (<0.6)')

        level_counts = Counter(confidence_levels)
        level_names = ['High (≥0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)']
        level_values = [level_counts.get(level, 0) for level in level_names]
        level_colors = ['#2ECC71', '#F39C12', '#E74C3C']  # Green, Orange, Red

        bars3 = ax3.bar(level_names, level_values,
                        color=level_colors, alpha=0.8)
        ax3.set_title('Confidence Level Distribution',
                      fontsize=self.font_size + 2, pad=20)
        ax3.set_ylabel('Number of Articles', fontsize=self.font_size)
        ax3.tick_params(axis='x', rotation=15, labelsize=self.font_size - 1)

        # Add count labels on bars
        for bar, count in zip(bars3, level_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{count}', ha='center', va='bottom', fontsize=self.font_size)

        # Panel 4 (Bottom-Right): Average Confidence by Category
        category_confidence = {}
        for cat, conf in zip(categories, confidences):
            if cat not in category_confidence:
                category_confidence[cat] = []
            category_confidence[cat].append(conf)

        # Calculate average confidence for each category
        avg_confidences = {cat: np.mean(confs)
                           for cat, confs in category_confidence.items()}

        # Sort categories by average confidence
        sorted_categories = sorted(
            avg_confidences.items(), key=lambda x: x[1], reverse=True)
        cat_names, avg_confs = zip(
            *sorted_categories) if sorted_categories else ([], [])

        if cat_names:  # Only plot if we have data
            bar_colors4 = [self.category_colors.get(
                cat, '#95A5A6') for cat in cat_names]
            bars4 = ax4.bar(cat_names, avg_confs, color=bar_colors4, alpha=0.8)
            ax4.set_title('Average Confidence by Category',
                          fontsize=self.font_size + 2, pad=20)
            ax4.set_ylabel('Average Confidence', fontsize=self.font_size)
            ax4.set_ylim(0, 1.0)
            ax4.tick_params(axis='x', rotation=45,
                            labelsize=self.font_size - 1)

            # Add confidence value labels on bars
            for bar, conf in zip(bars4, avg_confs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{conf:.3f}', ha='center', va='bottom', fontsize=self.font_size - 1)

            # Add sample size annotations
            for i, (cat, conf) in enumerate(sorted_categories):
                sample_size = len(category_confidence[cat])
                ax4.text(i, 0.05, f'n={sample_size}', ha='center', va='bottom',
                         fontsize=self.font_size - 2, alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Classification dashboard saved to {output_png}")

    def plot_classification_summary(self, articles_data, output_png='classification_summary.png'):
        """
        Creates a simplified classification summary plot for quick overview.

        Args:
            articles_data: List of article dictionaries containing classification results
            output_png: Output filename for the summary plot
        """
        categories = [article.get('category', 'unknown')
                      for article in articles_data]
        confidences = [float(article.get('classification_confidence', 0.0))
                       for article in articles_data]

        # Create summary statistics
        category_counts = Counter(categories)
        avg_confidence = np.mean(confidences) if confidences else 0
        high_conf_count = sum(1 for c in confidences if c >= 0.8)

        # Single plot summary
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Bar plot of category counts with confidence color coding
        cat_names = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = [self.category_colors.get(
            cat, '#95A5A6') for cat in cat_names]

        bars = ax.bar(cat_names, counts, color=colors, alpha=0.8)
        ax.set_title(f'Classification Results Summary\n'
                     f'Avg Confidence: {avg_confidence:.3f} | '
                     f'High Confidence: {high_conf_count}/{len(articles_data)} articles',
                     fontsize=self.font_size + 2)
        ax.set_ylabel('Number of Articles', fontsize=self.font_size)
        ax.set_xlabel('Category', fontsize=self.font_size)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom', fontsize=self.font_size)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Classification summary saved to {output_png}")
