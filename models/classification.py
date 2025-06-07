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

    # models/classification.py

    def classify(self, text, return_label=True, return_confidence=False):
        """
        Classify a single text using BERT with optional confidence scores.
        """
        if not text or not text.strip():
            if return_confidence:
                return "unknown", 0.0, {}
            return "unknown" if return_label else None

        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        if return_label:
            label = self.label_map.get(pred_idx, f"unknown_{pred_idx}")
            if return_confidence:
                # Return all class probabilities
                all_probs = {self.label_map.get(i, f"class_{i}"): float(probs[0][i].item())
                             for i in range(len(self.label_map))}
                return label, float(confidence), all_probs
            return label
        else:
            return logits.cpu().numpy()

    def classify_with_scores(self, text):
        """
        Classify text and return detailed scoring information.
        """
        try:
            label, confidence, all_probs = self.classify(
                text, return_confidence=True)

            # Sort probabilities by score
            sorted_probs = sorted(
                all_probs.items(), key=lambda x: x[1], reverse=True)

            return {
                'predicted_label': label,
                'confidence': confidence,
                'all_scores': all_probs,
                'top_3_predictions': sorted_probs[:3],
                'success': True
            }
        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'predicted_label': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'top_3_predictions': [],
                'success': False,
                'error': str(e)
            }

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
