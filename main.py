"""
main.py (FIXED - Enhanced with unique naming and complete pipeline support)
- FIXED: Plotting logic now matches actual generated summaries
- FIXED: Only plot methods that were actually used/generated
- Unique file names for JSON and PNG outputs
- Enhanced plotting titles and file organization
- Complete pipeline configuration support
- NEW: Added classification visualization dashboard
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
    print("==== RSS Summarization Pipeline (Enhanced Version) ====")
    rss = ask('Enter RSS feed URL or local file path', cast_type=str)
    max_articles = ask('Enter number of articles to process',
                       default=5, cast_type=int)
    pipeline_type = ask(
        'Pipeline type (pipeline/parallel/iterative)', default='pipeline', cast_type=str)

    # Initialize variables with defaults
    extractive = 'textrank'  # Default extractive method
    abstractive = 'bart'     # Default abstractive method
    combine = False
    parallel_mode = None

    if pipeline_type == 'pipeline':
        combine = yesno(
            'Combine all extractive summaries as prompt?', default='n')

    if pipeline_type == 'parallel':
        print('Parallel mode options:')
        print('  1. Single Extractive + Single Abstractive (1+1)')
        print('  2. All Extractive + Single Abstractive (3+1)')
        parallel_mode = ask(
            'Choose parallel mode (1+1, 3+1)', default='1+1', cast_type=str)

    # Prompt for extractive method only if needed
    if pipeline_type == 'pipeline' and combine:
        extractive = 'textrank'  # Will use all extractive methods when combine=True
        extractive_label = 'ALL'
    elif pipeline_type == 'pipeline' or (pipeline_type == 'parallel' and parallel_mode == '1+1'):
        extractive = ask('Extractive summarization method (textrank/lexrank/lsa)',
                         default='textrank', cast_type=str)
        extractive_label = extractive
    else:
        # For 3+1 mode, we don't need user to select extractive method
        extractive = 'textrank'  # Default value for parameter passing
        extractive_label = 'Auto-selected'

    # Advanced options
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

    # Run the selected pipeline with enhanced naming
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

        # Prepare plotting data for serial pipeline
        if combine:
            method_names = ['TextRank', 'LexRank', 'LSA',
                            abstractive.upper(), f'Hybrid({abstractive.upper()})']
            score_dicts = [
                result['average_scores']['textrank'],
                result['average_scores']['lexrank'],
                result['average_scores']['lsa'],
                result['average_scores']['abstractive'],
                result['average_scores']['hybrid']
            ]
            png_filename = f"serial_combined_{abstractive}_comparison.png"
            plot_title = f'Serial Pipeline: Combined Extractive + {abstractive.upper()}'

            # Define classification dashboard filename for combined serial pipeline
            classification_png = f"serial_combined_{abstractive}_classification.png"
            classification_title = f'Serial Pipeline Classification Analysis: Combined + {abstractive.upper()}'
        else:
            method_names = [extractive_label.upper(), abstractive.upper(
            ), f'{extractive_label.upper()}+{abstractive.upper()}']
            score_dicts = [
                result['average_scores']['extractive'],
                result['average_scores']['abstractive'],
                result['average_scores']['hybrid']
            ]
            png_filename = f"serial_{extractive}_{abstractive}_comparison.png"
            plot_title = f'Serial Pipeline: {extractive.upper()} + {abstractive.upper()}'

            # Define classification dashboard filename for regular serial pipeline
            classification_png = f"serial_{extractive}_{abstractive}_classification.png"
            classification_title = f'Serial Pipeline Classification Analysis: {extractive.upper()} + {abstractive.upper()}'

    elif pipeline_type == 'parallel':
        print(
            f"Running PARALLEL pipeline: mode={parallel_mode}, extractive={extractive}, abstractive={abstractive}")

        pipeline = ParallelPipeline(
            extractive_methods=['textrank', 'lexrank', 'lsa'],
            abstractive_methods=['bart', 't5', 'pegasus'],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )

        result = pipeline.run(
            rss,
            outdir=outdir,
            max_articles=max_articles,
            mode=parallel_mode,
            selected_extractive=extractive,
            selected_abstractive=abstractive
        )

        # FIXED: Prepare plotting data for parallel pipeline based on what was actually generated
        if parallel_mode == '1+1':
            method_names = [extractive.upper(), abstractive.upper(
            ), f'{extractive.upper()}+{abstractive.upper()}']
            score_dicts = [
                result['average_scores']['extractive'],
                result['average_scores']['abstractive'],
                result['average_scores']['combo']
            ]
            png_filename = f"parallel_1plus1_{extractive}_{abstractive}_comparison.png"
            plot_title = f'Parallel Pipeline (1+1): {extractive.upper()} + {abstractive.upper()}'

            # Define classification dashboard filename for 1+1 parallel pipeline
            classification_png = f"parallel_1plus1_{extractive}_{abstractive}_classification.png"
            classification_title = f'Parallel Pipeline (1+1) Classification: {extractive.upper()} + {abstractive.upper()}'
        elif parallel_mode == '3+1':
            method_names = ['TextRank', 'LexRank',
                            'LSA', abstractive.upper(), 'Hybrid']
            score_dicts = [
                result['average_scores']['textrank'],
                result['average_scores']['lexrank'],
                result['average_scores']['lsa'],
                result['average_scores']['abstractive'],
                result['average_scores']['best_single']
            ]
            png_filename = f"parallel_3plus1_all_{abstractive}_comparison.png"
            plot_title = f'Parallel Pipeline (3+1): All Extractive + {abstractive.upper()}'

            # Define classification dashboard filename for 3+1 parallel pipeline
            classification_png = f"parallel_3plus1_all_{abstractive}_classification.png"
            classification_title = f'Parallel Pipeline (3+1) Classification: All + {abstractive.upper()}'
        else:
            # Fallback for any other parallel mode
            method_names = [f'{extractive.upper()}+{abstractive.upper()}']
            score_dicts = [result['average_scores']['combo']]
            png_filename = f"parallel_{parallel_mode}_comparison.png"
            plot_title = f'Parallel Pipeline ({parallel_mode})'

            # Define classification dashboard filename for other parallel modes
            classification_png = f"parallel_{parallel_mode}_classification.png"
            classification_title = f'Parallel Pipeline ({parallel_mode}) Classification'

    elif pipeline_type == 'iterative':
        print(f"Running ITERATIVE pipeline: {abstractive}")

        pipeline = IterativePipeline(
            abstractive_method=abstractive,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            device=device if device not in (None, '', 'None') else None
        )
        result = pipeline.run(rss, outdir=outdir, max_articles=max_articles)

        # Prepare plotting data for iterative pipeline
        method_names = ['TextRank', 'LexRank', 'LSA',
                        abstractive.upper(), 'IterativeHybrid']
        score_dicts = [
            result['average_scores']['textrank'],
            result['average_scores']['lexrank'],
            result['average_scores']['lsa'],
            result['average_scores']['abstractive'],
            result['average_scores']['final']
        ]
        png_filename = f"iterative_refinement_{abstractive}_comparison.png"
        plot_title = f'Iterative Pipeline: Multi-stage Refinement with {abstractive.upper()}'

        # Define classification dashboard filename for iterative pipeline
        classification_png = f"iterative_refinement_{abstractive}_classification.png"
        classification_title = f'Iterative Pipeline Classification: {abstractive.upper()}'

    else:
        raise NotImplementedError(
            f"Pipeline type '{pipeline_type}' is not implemented!")

    # Generate and save the comparison plot with unique naming
    png_path = os.path.join(outdir, png_filename)
    plotter.plot_multi_panel(method_names, score_dicts,
                             output_png=png_path, title=plot_title)
    print(f"Comparison plot saved to {png_path}")

    # Generate classification analysis dashboard
    classification_path = os.path.join(outdir, classification_png)
    articles_data = result.get('articles', [])

    if articles_data:
        print(f"\nGenerating classification analysis dashboard...")
        plotter.plot_classification_dashboard(
            articles_data=articles_data,
            output_png=classification_path,
            title=classification_title
        )
        print(f"Classification dashboard saved to {classification_path}")

        # Also generate a simplified classification summary
        summary_png = classification_png.replace(
            '_classification.png', '_classification_summary.png')
        summary_path = os.path.join(outdir, summary_png)
        plotter.plot_classification_summary(
            articles_data=articles_data,
            output_png=summary_path
        )
        print(f"Classification summary saved to {summary_path}")
    else:
        print("Warning: No article data found for classification analysis")

    print("Pipeline execution completed successfully.")
    print(f"\nGenerated files:")
    print(f"  - Summarization comparison: {png_path}")
    if articles_data:
        print(f"  - Classification dashboard: {classification_path}")
        print(f"  - Classification summary: {summary_path}")


if __name__ == '__main__':
    main()
