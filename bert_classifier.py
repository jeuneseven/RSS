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
        self.labels = labels or [
            "Technology", "Business", "Science", "Entertainment", "Health"]
        if len(self.labels) != num_labels:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match num_labels ({num_labels})")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)}
        )
        self.model.to(self.device)

    def classify(self, text, threshold=0.3):
        if not text or len(text.strip()) == 0:
            return {"error": "No content available to classify"}

        inputs = self.tokenizer(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                logits, dim=1).cpu().numpy()[0]

        predictions = [
            {"label": self.labels[i], "confidence": float(prob)}
            for i, prob in enumerate(probabilities) if prob >= threshold
        ]

        if not predictions:
            max_idx = np.argmax(probabilities)
            predictions.append(
                {"label": self.labels[max_idx], "confidence": float(probabilities[max_idx])})

        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {"predictions": predictions, "top_category": predictions[0]["label"], "confidence": predictions[0]["confidence"]}


class ExtractiveSummarizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def summarize(self, text, top_n=3):
        sentences = split_into_sentences(text)
        if len(sentences) <= top_n:
            return {"summary": " ".join(sentences), "weights": [1.0] * len(sentences)}

        embeddings = self.model.encode(sentences)
        doc_embedding = np.mean(embeddings, axis=0)
        similarities = cosine_similarity([doc_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        summary_sentences = [sentences[i] for i in top_indices]

        # Normalize similarity scores
        weights = similarities[top_indices]
        norm_weights = (weights - weights.min()) / \
            (weights.max() - weights.min() + 1e-8)

        return {"summary": " ".join(summary_sentences), "weights": norm_weights.tolist(), "indices": top_indices.tolist()}


def classify_and_summarize_entries(feed_entries, custom_labels=None):
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
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        results = {"status": "success", "feed_title": feed.feed.get(
            'title', 'No title'), "entries": []}

        if classify:
            print("\nClassifying and summarizing articles...")
            processed = classify_and_summarize_entries(
                feed.entries, custom_labels)

            for entry in processed:
                print(f"\nTitle: {entry['title']}")
                print(
                    f"Top category: {entry['classification']['top_category']} (Confidence: {entry['classification']['confidence']:.4f})")
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
