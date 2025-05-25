"""
models/classification.py

Unified BERT-based text classification for RSS articles.
Uses Huggingface Transformers' BertTokenizer/BertForSequenceClassification.
Centralizes parameters to ensure consistency across pipelines.
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F


class BertClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=10, device=None):
        """
        Initialize BERT classifier for text classification tasks.
        :param model_name: Pretrained model name or path (default: 'bert-base-uncased')
        :param num_labels: Number of classes (default: 10)
        :param device: 'cuda' or 'cpu'. If None, auto-detect.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels).to(self.device)
        self.num_labels = num_labels
        # You should customize this externally
        self.label_map = {i: f"Label_{i}" for i in range(num_labels)}

    def classify(self, text, return_label=True):
        """
        Classify a single text using BERT.
        :param text: Text to classify
        :param return_label: If True, return label name, else return logits
        :return: Predicted label name or logits
        """
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        if return_label:
            return self.label_map.get(pred_idx, str(pred_idx))
        else:
            return logits.cpu().numpy()

    def batch_classify(self, texts, return_label=True):
        """
        Batch classify a list of texts.
        :param texts: List of text strings
        :param return_label: If True, return label names
        :return: List of predicted labels or logits
        """
        results = []
        for text in texts:
            results.append(self.classify(text, return_label=return_label))
        return results

    def set_label_map(self, label_map):
        """
        Set a custom mapping from class indices to label names.
        :param label_map: Dict (int->str)
        """
        self.label_map = label_map
