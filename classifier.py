from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import feedparser
import traceback
from utils import get_entry_content, split_into_sentences
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class BERTTextClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=5, labels=None):
        """
        Initialize BERT text classifier with custom labels
        Uses DistilBERT for efficient processing on CPU
        """
        self.labels = labels or [
            "Technology", "Business", "Science", "Entertainment", "Health"]
        if len(self.labels) != num_labels:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match num_labels ({num_labels})")

        # Force CPU usage for Intel i9 compatibility
        self.device = torch.device("cpu")  # Your Intel i9 doesn't have CUDA
        print(f"Using device: {self.device}")

        # Use a more lightweight model for CPU inference
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

    def classify(self, text, threshold=0.3):
        """
        Classify text into predefined categories using BERT

        Args:
            text (str): Text to classify
            threshold (float): Minimum confidence threshold for predictions

        Returns:
            dict: Classification results with predictions and confidence scores
        """
        if not text or len(text.strip()) == 0:
            return {"error": "No content available to classify"}

        # Tokenize input text
        inputs = self.tokenizer(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
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
            predictions.append(
                {"label": self.labels[max_idx], "confidence": float(probabilities[max_idx])})

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
    """

    def __init__(self):
        # Use a lightweight model for CPU inference
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def summarize(self, text, top_n=3):
        """
        Generate extractive summary using sentence embedding similarity

        Args:
            text (str): Text to summarize
            top_n (int): Number of sentences to extract

        Returns:
            dict: Summary information including selected sentences and weights
        """
        sentences = split_into_sentences(text)
        if len(sentences) <= top_n:
            return {"summary": " ".join(sentences), "weights": [1.0] * len(sentences)}

        # Generate embeddings for all sentences
        embeddings = self.model.encode(sentences)

        # Calculate document embedding as mean of sentence embeddings
        doc_embedding = np.mean(embeddings, axis=0)

        # Calculate similarity of each sentence to document embedding
        similarities = cosine_similarity([doc_embedding], embeddings)[0]

        # Select top N sentences
        top_indices = np.argsort(similarities)[-top_n:][::-1]
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


def classify_and_summarize_entries(feed_entries, custom_labels=None):
    """
    Process RSS feed entries with classification and summarization

    Args:
        feed_entries: List of feed entries
        custom_labels: Optional custom classification labels

    Returns:
        list: Processed entries with classification and summary
    """
    labels = custom_labels or ["Technology",
                               "Business", "Science", "Entertainment", "Health"]
    classifier = BERTTextClassifier(num_labels=len(labels), labels=labels)
    summarizer = ExtractiveSummarizer()

    results = []
    for entry in feed_entries:
        content = get_entry_content(entry)
        classification = classifier.classify(content)
        summary = summarizer.summarize(content)

        entry_result = {
            "title": entry.get("title", "No title"),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "classification": classification,
            "summary": summary
        }
        results.append(entry_result)

    return results


def process_rss_feed_with_classification_and_summary(file_path='rss.xml', classify=True, custom_labels=None):
    """
    Process RSS feed with classification and summarization

    Args:
        file_path (str): Path to RSS feed
        classify (bool): Whether to classify entries
        custom_labels (list): Optional custom labels

    Returns:
        dict: Processing results
    """
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        results = {
            "status": "success",
            "feed_title": feed.feed.get('title', 'No title'),
            "entries": []
        }

        if classify:
            print("\nClassifying and summarizing articles...")
            processed = classify_and_summarize_entries(
                feed.entries, custom_labels)

            for entry in processed:
                print(f"\nTitle: {entry['title']}")
                print(f"Top category: {entry['classification']['top_category']} "
                      f"(Confidence: {entry['classification']['confidence']:.4f})")
                print(f"Summary: {entry['summary']['summary']}")
                results["entries"].append(entry)
        else:
            for entry in feed.entries:
                results["entries"].append({
                    "title": entry.get('title', 'No title'),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', '')
                })

        return results

    except Exception as e:
        print(f"Error processing RSS: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    results = process_rss_feed_with_classification_and_summary()
    print(f"\nProcessed {len(results['entries'])} articles")
