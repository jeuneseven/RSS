import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Union, Optional

# Ensure common utilities are available
from utils import clean_html_text


class BERTTextClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 5, labels: Optional[List[str]] = None):
        """
        Initialize BERT text classifier with custom labels
        Uses DistilBERT for efficient processing on CPU

        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification labels
            labels: List of label names (defaults to common categories if None)
        """
        self.labels = labels or [
            "Technology", "Business", "Science", "Entertainment", "Health"]
        if len(self.labels) != num_labels:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match num_labels ({num_labels})")

        # Force CPU usage for Intel i9 compatibility
        self.device = torch.device("cpu")
        print(f"Using device: {self.device} for text classification")

        # Use a lightweight model for CPU inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)}
        )
        self.model.to(self.device)

        # Set model to inference mode
        self.model.eval()

    def classify(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Classify text into predefined categories using BERT

        Args:
            text: Text to classify
            threshold: Minimum confidence threshold for predictions

        Returns:
            Dictionary with classification results, predictions and confidence scores
        """
        if not text or len(text.strip()) == 0:
            return {"error": "No content available to classify"}

        # Clean the text first
        clean_text = clean_html_text(text)

        # Tokenize input text
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                logits, dim=1).cpu().numpy()[0]

        # Get predictions above threshold
        predictions = [
            {"label": self.labels[i], "confidence": float(prob)}
            for i, prob in enumerate(probabilities) if prob >= threshold
        ]

        # If no predictions meet threshold, use the highest confidence one
        if not predictions:
            max_idx = np.argmax(probabilities)
            predictions.append({
                "label": self.labels[max_idx],
                "confidence": float(probabilities[max_idx])
            })

        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "predictions": predictions,
            "top_category": predictions[0]["label"],
            "confidence": predictions[0]["confidence"]
        }


class ExtractiveSummarizer:
    """
    Extractive summarizer using sentence embeddings and cosine similarity
    This is an alternative summarization approach that can be used if sentence-transformers is installed
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a lightweight sentence transformer model

        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"Initialized ExtractiveSummarizer with model: {model_name}")
            self.initialized = True
        except ImportError:
            print(
                "Warning: sentence-transformers not installed. ExtractiveSummarizer will not function.")
            print("Install with: pip install sentence-transformers")
            self.initialized = False

    def summarize(self, text: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Generate extractive summary using sentence embedding similarity

        Args:
            text: Text to summarize
            top_n: Number of sentences to extract

        Returns:
            Dictionary with summary information including selected sentences and weights
        """
        if not self.initialized:
            return {"error": "SentenceTransformer not initialized. Install sentence-transformers package."}

        from sklearn.metrics.pairwise import cosine_similarity
        import nltk
        nltk.download('punkt', quiet=True)

        # Split text into sentences
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= top_n:
            return {
                "summary": text,
                "weights": [1.0] * len(sentences),
                "indices": list(range(len(sentences)))
            }

        # Generate embeddings for all sentences
        embeddings = self.model.encode(sentences)

        # Calculate document embedding as mean of sentence embeddings
        doc_embedding = np.mean(embeddings, axis=0)

        # Calculate similarity of each sentence to document embedding
        similarities = cosine_similarity([doc_embedding], embeddings)[0]

        # Select top N sentences
        top_indices = np.argsort(similarities)[-top_n:]

        # Sort indices by original order to maintain narrative flow
        top_indices.sort()

        summary_sentences = [sentences[i] for i in top_indices]

        # Normalize similarity scores to weights
        weights = similarities[top_indices]
        norm_weights = (weights - weights.min()) / \
            (weights.max() - weights.min() + 1e-8)

        return {
            "summary": " ".join(summary_sentences),
            "weights": norm_weights.tolist(),
            "indices": top_indices.tolist()
        }
