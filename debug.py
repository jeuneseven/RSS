import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('punkt')


def textrank_summary(text, sent_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sent_count)
    return " ".join([str(sentence) for sentence in summary])


def bart_summary(text, max_len=130, min_len=40):
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.batch_encode_plus(
        [text], return_tensors='pt', truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs['input_ids'], num_beams=4, max_length=max_len, min_length=min_len, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 用BART为各部分生成短语


def generate_phrases_by_section(text, section_len=2):
    sentences = nltk.sent_tokenize(text)
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    phrases = []
    for i in range(0, len(sentences), section_len):
        part = " ".join(sentences[i:i+section_len])
        if not part.strip():
            continue
        inputs = tokenizer.batch_encode_plus(
            [part], return_tensors='pt', truncation=True, max_length=256)
        summary_ids = model.generate(
            inputs['input_ids'], num_beams=4, max_length=16, min_length=3, early_stopping=True)
        phrase = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        phrases.append(phrase)
    return phrases

# 用BERTScore计算每个短语与所有句子的相似度，选出最相关句子


def phrase_guided_extraction(text, phrases, top_k=4):
    sentences = nltk.sent_tokenize(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sent_emb = model.encode(sentences)
    phrase_emb = model.encode(phrases)
    selected = set()
    scores = []
    for emb in phrase_emb:
        sims = [np.dot(emb, s) / (np.linalg.norm(emb) *
                                  np.linalg.norm(s) + 1e-8) for s in sent_emb]
        idx = int(np.argmax(sims))
        selected.add(idx)
        scores.append((idx, sims[idx]))
    # 如果不足top_k，则按和所有phrase平均相似度从高到低补齐
    if len(selected) < top_k:
        avg_sims = np.mean([
            [np.dot(emb, s) / (np.linalg.norm(emb)*np.linalg.norm(s) + 1e-8)
             for emb in phrase_emb]
            for s in sent_emb
        ], axis=1)
        for idx in np.argsort(avg_sims)[::-1]:
            if len(selected) >= top_k:
                break
            selected.add(idx)
    selected = sorted(selected)
    return " ".join([sentences[i] for i in selected])


def evaluate(summary, reference):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    P, R, F1 = bert_score([summary], [reference], lang="en", verbose=False)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "bertscore": float(F1.mean())
    }


def percent_improve(new, base):
    return 100 * (new - base) / (base + 1e-8)


if __name__ == "__main__":
    original_text = """We’re coming up on the 10-year anniversary of Undertale ’s release, and to mark the occasion, a 25-piece orchestra will perform the game’s soundtrack for a one-night-only concert at London's Eventim Apollo this summer. The event, dubbed The Determination Symphony , will be held on June 22, and tickets are on sale now. The Determination Symphony is described as “a musical journey from your initial fall at Mount Ebott, leading you through Froggit Village, the Snowdin Forest, Temmie Village and so much more.” Attendees (who I’m deeply envious of) will be able to watch all of that on screen while the orchestra makes its way through arrangements of the soundtrack. It’s hard to believe that Toby Fox’s Undertale is already 10 years old, but its enduring popularity just speaks to the impact it’s had on so many who have played it. We may not all get to experience the orchestral rendition, but at least we'll always have the original soundtrack . This article originally appeared on Engadget at https://www.engadget.com/gaming/a-live-orchestra-will-perform-undertales-soundtrack-in-london-to-celebrate-its-10th-anniversary-214830716.html?src=rss"""

    tr_sum = textrank_summary(original_text, sent_count=3)
    bart_sum = bart_summary(original_text, max_len=80, min_len=30)
    phrases = generate_phrases_by_section(original_text, section_len=2)
    fusion_sum = phrase_guided_extraction(original_text, phrases, top_k=4)

    score_tr = evaluate(tr_sum, original_text)
    score_bart = evaluate(bart_sum, original_text)
    score_fusion = evaluate(fusion_sum, original_text)

    print("\n==== TextRank ====")
    print(score_tr)
    print("\n==== BART ====")
    print(score_bart)
    print("\n==== Phrases (from BART): ====")
    print(phrases)
    print("\n==== Fusion (Phrase-guided extraction) ====")
    print(fusion_sum)
    print(score_fusion)

    print("\n==== Improvements Over Baselines ====")
    for metric in ["rouge1", "rouge2", "rougeL", "bertscore"]:
        tr_improve = percent_improve(score_fusion[metric], score_tr[metric])
        bart_improve = percent_improve(
            score_fusion[metric], score_bart[metric])
        print(
            f"{metric.upper()}: vs TextRank: {tr_improve:+.2f}%  vs BART: {bart_improve:+.2f}%")
