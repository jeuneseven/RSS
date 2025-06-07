"""
models/pretrained_classifier.py

Simplified pre-trained classification using zero-shot learning.
"""

from transformers import pipeline
import torch


class PretrainedClassifier:
    def __init__(self, device=None):
        """
        Initialize zero-shot classifier for news categorization.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print("🔄 Loading zero-shot classifier...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == 'cuda' else -1
            )
            print("✅ Zero-shot classifier loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load zero-shot classifier: {e}")
            raise

        # Define our target categories with descriptive names for better classification
        self.categories = [
            "technology and science",
            "sports and athletics",
            "politics and government",
            "business and finance",
            "entertainment and culture",
            "health and medicine"
        ]

        # Mapping to simplified labels
        self.label_mapping = {
            "technology and science": "technology",
            "sports and athletics": "sports",
            "politics and government": "politics",
            "business and finance": "business",
            "entertainment and culture": "entertainment",
            "health and medicine": "health"
        }

    def classify(self, text):
        """
        Classify text using zero-shot classification.
        """
        if not text or not text.strip():
            return "unknown"

        try:
            # Truncate text for better performance
            text_truncated = ' '.join(text.split()[:300])

            result = self.classifier(text_truncated, self.categories)

            # Get the best prediction
            best_category = result['labels'][0]
            confidence = result['scores'][0]

            # Map to simplified label
            simplified_label = self.label_mapping.get(best_category, "unknown")

            return simplified_label

        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown"

    def classify_with_scores(self, text):
        """
        Classify text and return detailed scoring information.
        """
        if not text or not text.strip():
            return {
                'predicted_label': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'top_3_predictions': [],
                'success': False,
                'method': 'zero-shot'
            }

        try:
            # Truncate text for better performance
            text_truncated = ' '.join(text.split()[:300])

            result = self.classifier(text_truncated, self.categories)

            # Get the best prediction
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            predicted_label = self.label_mapping.get(best_category, "unknown")

            # Create all scores dict with simplified labels
            all_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                simplified = self.label_mapping.get(label, label)
                all_scores[simplified] = score

            # Create top 3 predictions with simplified labels
            top_3 = []
            for label, score in zip(result['labels'][:3], result['scores'][:3]):
                simplified = self.label_mapping.get(label, label)
                top_3.append((simplified, score))

            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_scores': all_scores,
                'top_3_predictions': top_3,
                'success': True,
                'method': 'zero-shot'
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
                'error': str(e),
                'method': 'zero-shot'
            }
