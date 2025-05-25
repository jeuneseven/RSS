"""
main.py (combine as prompt moved before method selection, interaction more logical)

- Now asks 'Combine all extractive summaries as prompt?' immediately after pipeline type
- Combine flag is only relevant for serial pipeline
- Parallel/iterative structures skip this prompt
- Only then proceed to method selection and advanced params
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

    # Combine prompt (only relevant for serial pipeline)
    if pipeline_type == 'pipeline':
        combine = yesno(
            'Combine all extractive summaries as prompt?', default='n')
    else:
        combine = False

    # Parallel mode if needed
    if pipeline_type == 'parallel':
        print('Parallel mode supports:')
        print('  1. Single Extractive Combine with Single Abstractive (1+1)')
        print('  2. Best Extractive Combine with Single Abstractive (3+1)')
        print('  3. Best Extractive Combine with Best Abstractive (3+3)')
        parallel_mode = ask(
            'Choose parallel mode (1+1, 3+1, 3+3)', default='1+1', cast_type=str)
    else:
        parallel_mode = None

    if yesno('Show advanced options?'):
        extractive = ask('Extractive summarization method (textrank/lexrank/lsa)',
                         default='textrank', cast_type=str)
        abstractive = ask('Abstractive model (bart/t5/pegasus)',
                          default='bart', cast_type=str)
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
        extractive = 'textrank'
        abstractive = 'bart'
        outdir = 'data/outputs/'
        max_length = 150
        min_length = 50
        num_beams = 4
        device = None
    ensure_dir(outdir)

    # Pipeline logic
    if pipeline_type == 'pipeline':
        print(
            f"Running SERIAL pipeline: {extractive} -> {abstractive} (combine={combine})")
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
        methods = ['Extractive', 'Abstractive', 'Hybrid']
        avg_scores = [
            result['average_scores']['extractive'],
            result['average_scores']['abstractive'],
            result['average_scores']['hybrid']
        ]
    elif pipeline_type == 'parallel':
        if parallel_mode == '1+1':
            extractive_methods = [extractive]
            abstractive_methods = [abstractive]
        elif parallel_mode == '3+1':
            extractive_methods = ['textrank', 'lexrank', 'lsa']
            abstractive_methods = [abstractive]
        elif parallel_mode == '3+3':
            extractive_methods = ['textrank', 'lexrank', 'lsa']
            abstractive_methods = ['bart', 't5', 'pegasus']
        else:
            extractive_methods = [extractive]
            abstractive_methods = [abstractive]
        print(f"Running PARALLEL pipeline: {parallel_mode}")
        pipeline = ParallelPipeline(
            extractive_methods=extractive_methods,
            abstractive_methods=abstractive_methods,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir, max_articles=max_articles)
        methods = ['SingleExt+SingleAbs',
                   'BestExt+SingleAbs', 'BestExt+BestAbs']
        avg_scores = [
            result['average_scores']['combo'],
            result['average_scores']['best_single'],
            result['average_scores']['best_best']
        ]
    elif pipeline_type == 'iterative':
        print(f"Running ITERATIVE pipeline...")
        pipeline = IterativePipeline(
            abstractive_method=abstractive,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir, max_articles=max_articles)
        methods = ['Iterative-Final']
        avg_scores = [result['average_scores']['final']]
    else:
        raise NotImplementedError(
            f"Pipeline type '{pipeline_type}' is not implemented!")
    # --- Visualization ---
    plotter = Plotter()
    png_path = os.path.join(
        outdir, f"{pipeline_type}_{extractive}_{abstractive}_comparison.png")
    plotter.plot_bar(methods, avg_scores, output_png=png_path,
                     title='Summary Methods Comparison')
    print(f"PNG plot saved to {png_path}\nDone.")


if __name__ == '__main__':
    main()
