import nltk
import torch
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

# ---------------------- Summarizers ----------------------


def textrank_summary(text, num_sentences=5):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return [str(sentence) for sentence in summary]


def bart_summary(text, max_length=130):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn")
    model = model.to(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    inputs = tokenizer([text], max_length=1024,
                       return_tensors="pt", truncation=True)
    inputs = inputs.to(model.device)
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---------------------- Fusion Methods ----------------------


def interleaved_fusion(summ1, summ2):
    interleaved = []
    for s1, s2 in zip(summ1, summ2):
        interleaved.extend([s1, s2])
    interleaved += summ1[len(summ2):] + summ2[len(summ1):]
    return interleaved


def ranked_selection_fusion(summ1, summ2, original_text, top_k=5):
    all_sents = summ1 + summ2
    vectorizer = TfidfVectorizer().fit([original_text] + all_sents)
    sims = cosine_similarity(vectorizer.transform(
        all_sents), vectorizer.transform([original_text])).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [all_sents[i] for i in top_indices]


def weighted_score_fusion(summ1, summ2, weight1=0.6, weight2=0.4, top_k=5):
    all_sents = summ1 + summ2
    tfidf = TfidfVectorizer().fit(all_sents)
    tfidf_matrix = tfidf.transform(all_sents).toarray()
    scores = []
    for i, vec in enumerate(tfidf_matrix):
        w = weight1 if i < len(summ1) else weight2
        scores.append(w * np.sum(vec))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [all_sents[i] for i in top_indices]


def semantic_cluster_fusion(summ1, summ2, num_clusters=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sents = summ1 + summ2
    embeddings = model.encode(sents)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    selected = []
    for i in range(num_clusters):
        idx = np.argmin(np.linalg.norm(
            embeddings[kmeans.labels_ == i] - cluster_centers[i], axis=1))
        sent_idx = np.where(kmeans.labels_ == i)[0][idx]
        selected.append(sents[sent_idx])
    return selected

# ---------------------- Evaluation ----------------------


def evaluate_summary(summary, reference):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    bert_p, bert_r, bert_f1 = bert_score(
        [summary], [reference], lang="en", verbose=False)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
        "bertscore": round(float(bert_f1[0]), 4)
    }

# ---------------------- Main Script ----------------------


if __name__ == "__main__":
    text = """After abruptly pulling their support from what would have been the Senate’s first stablecoin regulatory bill, Senate Democrats announced Tuesday that they would introduce a new bill that would prevent federal officials and their families from issuing digital assets – a bill directed at Donald Trump and his family’s current stablecoin and meme coin holdings. “Currently, people who wish to cultivate influence with the president can enrich him personally by buying cryptocurrency he owns or controls,” said Sen. Jeff Merkley (D-OR), who introduced the bill to the Senate floor, in a press release . “This is a profoundly corrupt scheme. It endangers our national security and erodes public trust in government. Let’s end this corruption immediately.” The End Crypto Corruption Act comes in response to concerns inside the Democratic party that the GENIUS Act, which previously had strong bipartisan support, was inadequate at preventing corruption. Though the Senate Banking Committee passed the bill in March with a bipartisan vote, two subsequent developments reportedly pushed the Democrats to change course . First, a New York Times report last week revealed that the Trump family could potentially earn $2 billion from a stablecoin transaction with a Dubai-based investment firm under the current regulatory framework. Second, Trump announced a contest in April wherein the top holders of his meme coin would win a private dinner with the president, and the top 25 holders would win an additional guided tour of the White House. According to a report from Chainalysis, the meme coin’s issuers, Official Trump, have earned $320 million from trading fees from the contest alone. Though they admitted that there’s not much they can do to stop the president right now (see: no laws), Senate Republicans also expressed skepticism over the $TRUMP contest to NBC , and at least one staunch Trump ally, Sen. Cynthia Lummis of Wyoming, offered to partner with the Democrats on efforts to regulate lawmakers holding digital assets. “Even what may appear to be ‘cringey’ with regard to meme coins, it’s legal, and what we need to do is have a regulatory framework that makes this more clear, so we don’t have this Wild West scenario,” she told NBC."""

    textrank_sents = textrank_summary(text, num_sentences=5)
    bart_sents = sent_tokenize(bart_summary(text, max_length=130))

    fusion_results = {
        "TextRank": evaluate_summary(" ".join(textrank_sents), text),
        "Interleaved": evaluate_summary(" ".join(interleaved_fusion(textrank_sents, bart_sents)), text),
        "RankedSelection": evaluate_summary(" ".join(ranked_selection_fusion(textrank_sents, bart_sents, text)), text),
        "WeightedScore": evaluate_summary(" ".join(weighted_score_fusion(textrank_sents, bart_sents)), text),
        "SemanticCluster": evaluate_summary(" ".join(semantic_cluster_fusion(textrank_sents, bart_sents)), text)
    }

    from pprint import pprint
    pprint(fusion_results)
