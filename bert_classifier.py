from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class BERTTextClassifier:
    """
    A text classifier using BERT model for categorizing RSS feed content
    """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=5, labels=None):
        """
        Initialize the BERT text classifier

        Args:
            model_name (str): The pre-trained model to use
            num_labels (int): Number of classification categories
            labels (list): List of label names. If None, will use default labels
        """
        self.labels = labels or [
            "Technology", "Business", "Science", "Entertainment", "Health"]
        if len(self.labels) != num_labels:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match num_labels ({num_labels})")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)}
        )
        self.model.to(self.device)

    def classify(self, text, threshold=0.3):
        """
        Classify the input text into predefined categories

        Args:
            text (str): The text to classify
            threshold (float): Confidence threshold for multi-label classification

        Returns:
            dict: Dictionary with predicted labels and confidence scores
        """
        # Clean and prepare the text
        if not text or len(text.strip()) == 0:
            return {"error": "No content available to classify"}

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                logits, dim=1).cpu().numpy()[0]

        # Get predictions with confidence scores
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob >= threshold:
                predictions.append({
                    "label": self.labels[i],
                    "confidence": float(prob)
                })

        # Sort predictions by confidence (highest first)
        predictions = sorted(
            predictions, key=lambda x: x["confidence"], reverse=True)

        # If no category meets the threshold, return the highest one
        if not predictions:
            max_idx = np.argmax(probabilities)
            predictions.append({
                "label": self.labels[max_idx],
                "confidence": float(probabilities[max_idx])
            })

        return {
            "predictions": predictions,
            "top_category": predictions[0]["label"],
            "confidence": predictions[0]["confidence"]
        }

    def fine_tune(self, texts, labels, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tune the BERT model on custom data

        Args:
            texts (list): List of text samples
            labels (list): List of corresponding labels (as integers)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for the optimizer

        Returns:
            dict: Training statistics
        """
        from torch.utils.data import Dataset, DataLoader
        from transformers import AdamW

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True, max_length=max_length)
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx])
                        for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        # Prepare dataset
        dataset = TextDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        self.model.eval()
        return {"average_loss": total_loss / epochs}


def bert_classification(text, model_name="distilbert-base-uncased", custom_labels=None):
    """
    Classify text using BERT model

    Args:
        text (str): The text to classify
        model_name (str): The pre-trained model to use
        custom_labels (list): Optional custom label set

    Returns:
        dict: Classification results
    """
    # Define default or use custom labels
    labels = custom_labels or ["Technology",
                               "Business", "Science", "Entertainment", "Health"]

    # Initialize classifier
    classifier = BERTTextClassifier(
        model_name=model_name, num_labels=len(labels), labels=labels)

    # Classify the text
    results = classifier.classify(text)

    return results


def classify_rss_entries(feed_entries, custom_labels=None):
    """
    Classify multiple RSS entries

    Args:
        feed_entries (list): List of RSS feed entries
        custom_labels (list): Optional custom label set

    Returns:
        list: List of entries with classification results added
    """
    # Initialize the classifier only once for efficiency
    labels = custom_labels or ["Technology",
                               "Business", "Science", "Entertainment", "Health"]
    classifier = BERTTextClassifier(num_labels=len(labels), labels=labels)

    classified_entries = []

    for entry in feed_entries:
        # Get content from the entry
        content = get_entry_content(entry)

        # Classify the content
        classification = classifier.classify(content)

        # Add classification to the entry
        entry_with_classification = entry.copy()
        entry_with_classification['classification'] = classification

        classified_entries.append(entry_with_classification)

    return classified_entries


# Example of how to use these functions with your existing RSS processor
def process_rss_feed_with_classification(file_path='rss.xml', classify=True):
    """
    Process an RSS feed with text classification

    Args:
        file_path (str): Path to the RSS feed file
        classify (bool): Whether to classify the entries

    Returns:
        dict: Processing results
    """
    try:
        print(f"Parsing RSS file: {file_path}")
        feed = feedparser.parse(file_path)

        # Print basic information
        print(f"Feed title: {feed.feed.get('title', 'No title')}")
        print(f"Found {len(feed.entries)} articles")

        # Check if there are entries
        if len(feed.entries) == 0:
            print("No articles found.")
            return {"status": "error", "message": "No articles found"}

        results = {"status": "success", "feed_title": feed.feed.get(
            'title', 'No title'), "entries": []}

        # Process entries
        if classify:
            print("\nClassifying articles...")
            classified_entries = classify_rss_entries(feed.entries)

            for entry in classified_entries:
                print(f"\nTitle: {entry.get('title', 'No title')}")
                print(
                    f"Classification: {entry['classification']['top_category']} (Confidence: {entry['classification']['confidence']:.4f})")
                print(
                    f"All categories: {', '.join([f'{p['label']} ({p['confidence']:.2f})' for p in entry['classification']['predictions']])}")

                results["entries"].append({
                    "title": entry.get('title', 'No title'),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', ''),
                    "classification": entry['classification']
                })
        else:
            # Just process without classification
            for entry in feed.entries:
                results["entries"].append({
                    "title": entry.get('title', 'No title'),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', '')
                })

        return results

    except Exception as e:
        print(f"Error processing RSS: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
