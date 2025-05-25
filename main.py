"""
main.py (parallel 3+1/3+3 不再提示 extractive, 只有 1+1 才提示)
- Only prompt for extractive if mode is 1+1 in parallel, otherwise skip
- All logic for serial/iterative/parallel auto adapts
"""

import os
from utils.common import ensure_dir
from pipelines.pipeline_serial import SerialPipeline
from pipelines.pipeline_parallel import ParallelPipeline
from pipelines.pipeline_iterative import IterativePipeline
from evaluation.plotter import Plotter


def ask(prompt, default=None, cast_type=str, allow_blank=False):
    while True:
        s = input(
            f"{prompt}" + (f" (default: {default}): " if default is not None else ': '))
        if not s.strip():
            if default is not None or allow_blank:
                return default
            else:
                continue
        try:
            return cast_type(s)
        except Exception:
            print(f"Invalid input! Expecting {cast_type.__name__}.")


def yesno(prompt, default='n'):
    s = input(f"{prompt} (y/n, default: {default}): ").strip().lower()
    return (s == 'y') or (s == '' and default == 'y')


def main():
    print("==== RSS Summarization Pipeline (Stepwise) ====")
    rss = ask('Enter RSS feed URL or local file path', cast_type=str)
    max_articles = ask('Enter number of articles to process',
                       default=5, cast_type=int)
    pipeline_type = ask(
        'Pipeline type (pipeline/parallel/iterative)', default='pipeline', cast_type=str)
    if pipeline_type == 'pipeline':
        combine = yesno(
            'Combine all extractive summaries as prompt?', default='n')
    else:
        combine = False
    if pipeline_type == 'parallel':
        print('Parallel mode options:')
        print('  1. Single Extractive + Single Abstractive (1+1)')
        print('  2. Best Extractive + Single Abstractive (3+1)')
        print('  3. Best Extractive + Best Abstractive (3+3)')
        parallel_mode = ask(
            'Choose parallel mode (1+1, 3+1, 3+3)', default='1+1', cast_type=str)
    else:
        parallel_mode = None
    # Prompt for extractive only if needed
    if pipeline_type == 'pipeline' and combine:
        extractive = 'textrank'
        extractive_label = 'ALL'
    elif pipeline_type == 'pipeline' or (pipeline_type == 'parallel' and parallel_mode == '1+1'):
        extractive = ask('Extractive summarization method (textrank/lexrank/lsa)',
                         default='textrank', cast_type=str)
        extractive_label = extractive
    else:
        extractive = None
        extractive_label = None
    abstractive = ask('Abstractive model (bart/t5/pegasus)',
                      default='bart', cast_type=str)
    if yesno('Show other advanced options?', default='n'):
        outdir = ask('Output directory',
                     default='data/outputs/', cast_type=str)
        max_length = ask('Maximum summary length (tokens)',
                         default=150, cast_type=int)
        min_length = ask('Minimum summary length (tokens)',
                         default=50, cast_type=int)
        num_beams = ask('Number of beams for generation',
                        default=4, cast_type=int)
        device = ask('Device (cuda/cpu/None)', default=None,
                     cast_type=str, allow_blank=True)
    else:
        outdir = 'data/outputs/'
        max_length = 150
        min_length = 50
        num_beams = 4
        device = None
    ensure_dir(outdir)
    plotter = Plotter()
    if pipeline_type == 'pipeline':
        print(
            f"Running SERIAL pipeline: {extractive_label} -> {abstractive} (combine={combine})")
        pipeline = SerialPipeline(
            extractive_method=extractive,
            abstractive_method=abstractive,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir,
                              max_articles=max_articles, combine=combine)
        if combine:
            method_names = ['TextRank', 'LexRank', 'LSA',
                            abstractive, f'Hybrid({abstractive})']
            score_dicts = [
                result['average_scores']['textrank'],
                result['average_scores']['lexrank'],
                result['average_scores']['lsa'],
                result['average_scores']['abstractive'],
                result['average_scores']['hybrid']
            ]
        else:
            method_names = [extractive_label, abstractive,
                            f'{extractive_label}+{abstractive}']
            score_dicts = [
                result['average_scores']['extractive'],
                result['average_scores']['abstractive'],
                result['average_scores']['hybrid']
            ]
    elif pipeline_type == 'parallel':
        pipeline = ParallelPipeline(
            extractive_methods=['textrank', 'lexrank', 'lsa'],
            abstractive_methods=['bart', 't5', 'pegasus'],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir,
                              max_articles=max_articles, mode=parallel_mode)
        if parallel_mode == '1+1':
            method_names = [extractive, abstractive,
                            f'{extractive}+{abstractive}']
            score_dicts = [
                result['average_scores']['extractive'],
                result['average_scores']['abstractive'],
                result['average_scores']['combo']
            ]
        elif parallel_mode == '3+1':
            method_names = ['TextRank', 'LexRank',
                            'LSA', abstractive, 'Hybrid']
            score_dicts = [
                result['average_scores']['textrank'],
                result['average_scores']['lexrank'],
                result['average_scores']['lsa'],
                result['average_scores']['abstractive'],
                result['average_scores']['best_single']
            ]
        elif parallel_mode == '3+3':
            method_names = ['BestExtractive', 'BestAbstractive', 'Hybrid']
            score_dicts = [
                result['average_scores']['best_extract'],
                result['average_scores']['best_abstractive'],
                result['average_scores']['best_best']
            ]
        else:
            method_names = [f'{extractive}+{abstractive}']
            score_dicts = [result['average_scores']['combo']]
    elif pipeline_type == 'iterative':
        pipeline = IterativePipeline(
            abstractive_method=abstractive,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir, max_articles=max_articles)
        method_names = ['TextRank', 'LexRank', 'LSA', abstractive, 'Hybrid']
        score_dicts = [
            result['average_scores']['textrank'],
            result['average_scores']['lexrank'],
            result['average_scores']['lsa'],
            result['average_scores']['abstractive'],
            result['average_scores']['final']
        ]
    else:
        raise NotImplementedError(
            f"Pipeline type '{pipeline_type}' is not implemented!")
    png_path = os.path.join(outdir, f"{pipeline_type}_comparison.png")
    plotter.plot_multi_panel(method_names, score_dicts,
                             output_png=png_path, title='Summarization Methods Comparison')
    print(f"PNG plot saved to {png_path}\nDone.")


if __name__ == '__main__':
    main()
