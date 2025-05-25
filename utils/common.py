"""
utils/common.py

Common utility functions and global configuration management.
- Loads YAML/JSON config files for experiment reproducibility
- Provides argument parsing for command line pipelines
- Miscellaneous helpers for text, file, or environment handling
"""

import json
import yaml
import argparse
import os


class Config:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path):
        """
        Load configuration from YAML or JSON file.
        """
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError('Unsupported config file type!')
        return self.config

    def get(self, key, default=None):
        return self.config.get(key, default)


def add_pipeline_args(parser):
    """
    Add common pipeline arguments for command line usage.
    :param parser: argparse.ArgumentParser
    """
    parser.add_argument('--rss', type=str, required=True,
                        help='RSS feed URL or local XML path')
    parser.add_argument('--outdir', type=str,
                        default='data/outputs/', help='Output directory')
    parser.add_argument('--max_articles', type=int, default=5,
                        help='Maximum number of articles to process')
    parser.add_argument('--extractive', type=str, default='textrank', choices=[
                        'textrank', 'lexrank', 'lsa'], help='Extractive summarization method')
    parser.add_argument('--abstractive', type=str, default='bart',
                        choices=['bart', 't5', 'pegasus'], help='Abstractive summarization model')
    parser.add_argument('--hybrid', type=str, default='none', choices=[
                        'none', 'pipeline', 'parallel', 'iterative'], help='Hybrid pipeline structure')
    parser.add_argument('--max_length', type=int, default=150,
                        help='Maximum summary length (tokens)')
    parser.add_argument('--min_length', type=int, default=50,
                        help='Minimum summary length (tokens)')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for generation')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda/cpu (auto if None)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config YAML/JSON for all params')
    return parser


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, filepath):
    """Save object as JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
