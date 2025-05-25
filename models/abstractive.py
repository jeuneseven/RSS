"""
models/abstractive.py

Unified interface for abstractive summarization algorithms: BART, T5, Pegasus
All parameters are managed centrally for consistency.
Algorithms are called as black boxes using the Huggingface transformers library.
"""

from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer
)
import torch


class AbstractiveSummarizer:
    def __init__(self, max_length=150, min_length=50, num_beams=4, length_penalty=2.0, device=None):
        """
        AbstractiveSummarizer manages all abstractive summarization models with unified parameters.
        :param max_length: Maximum summary length in tokens (default: 150)
        :param min_length: Minimum summary length in tokens (default: 50)
        :param num_beams: Beam search width (default: 4)
        :param length_penalty: Length penalty for beam search (default: 2.0)
        :param device: 'cuda' or 'cpu'. If None, auto-detect.
        """
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load models and tokenizers only once per instance
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn').to(self.device)

        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            't5-base').to(self.device)

        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(
            'google/pegasus-xsum')
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
            'google/pegasus-xsum').to(self.device)

    def bart(self, text):
        """
        BART abstractive summarization.
        :param text: Article text
        :return: Summarized text (string)
        """
        inputs = self.bart_tokenizer(
            text, truncation=True, max_length=1024, return_tensors="pt"
        ).to(self.device)
        summary_ids = self.bart_model.generate(
            inputs['input_ids'],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            early_stopping=True
        )
        summary = self.bart_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary

    def t5(self, text):
        """
        T5 abstractive summarization.
        :param text: Article text
        :return: Summarized text (string)
        """
        input_text = "summarize: " + text
        inputs = self.t5_tokenizer(
            input_text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        summary_ids = self.t5_model.generate(
            inputs['input_ids'],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            early_stopping=True
        )
        summary = self.t5_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary

    def pegasus(self, text):
        """
        Pegasus abstractive summarization.
        :param text: Article text
        :return: Summarized text (string)
        """
        inputs = self.pegasus_tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        summary_ids = self.pegasus_model.generate(
            inputs['input_ids'],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            early_stopping=True
        )
        summary = self.pegasus_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary
