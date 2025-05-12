from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import numpy as np

nltk.download("punkt")


def textrank_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(X)
    scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i]
                        for i in np.argsort(scores)[-num_sentences:]]
    ranked_sentences.sort(key=lambda s: sentences.index(s))
    return ' '.join(ranked_sentences)


def evaluate_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(p, r) for p, r in zip(preds, refs)]
    return {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores]),
    }


datasets_info = {
    "xsum": {"name": "xsum", "doc": "document", "ref": "summary"},
    "cnn": {"name": "cnn_dailymail", "doc": "article", "ref": "highlights", "config": "3.0.0"}
}

for dataset_key, info in datasets_info.items():
    print(f"\n=== Evaluating {dataset_key.upper()} ===")
    dataset = load_dataset(
        info["name"], name=info.get("config"), split="test[:5]")
    documents = [item[info["doc"]] for item in dataset]
    references = [item[info["ref"]] for item in dataset]

    summaries = [textrank_summary(doc) for doc in documents]
    rouge = evaluate_rouge(summaries, references)
    P, R, F1 = bert_score(summaries, references, lang="en", verbose=False)

    print("ROUGE-1:", round(rouge['rouge1'], 4))
    print("ROUGE-2:", round(rouge['rouge2'], 4))
    print("ROUGE-L:", round(rouge['rougeL'], 4))
    print("BERTScore-F1:", round(F1.mean().item(), 4))
