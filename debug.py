import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
import torch
import warnings
warnings.filterwarnings('ignore')


class HybridSummaryEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize BART model
        print("Loading BART model...")
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model.to(self.device)

        # Initialize TextRank
        self.textrank_summarizer = TextRankSummarizer(Stemmer('english'))

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        print("Initialization complete!")

    def textrank_summary(self, text, sentence_count=5):
        """使用TextRank提取式摘要"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer('english'))
            sentences = self.textrank_summarizer(
                parser.document, sentence_count)
            summary = ' '.join([str(sentence) for sentence in sentences])
            return summary
        except Exception as e:
            print(f"TextRank error: {e}")
            # 如果失败，返回前几句作为fallback
            sentences = text.split('.')[:sentence_count]
            return '. '.join(sentences) + '.'

    def bart_summary(self, text, max_length=150, min_length=50):
        """使用BART生成式摘要"""
        try:
            # 截断过长的文本
            if len(text) > 1024:
                text = text[:1024]

            inputs = self.bart_tokenizer.encode(text, return_tensors='pt',
                                                max_length=1024, truncation=True)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                summary_ids = self.bart_model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = self.bart_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"BART error: {e}")
            return "Summary generation failed."

    def generate_hybrid_candidates(self, text, num_candidates=5):
        """生成多个混合摘要候选"""
        candidates = []

        # 获取TextRank句子（不同数量）
        for sent_count in [3, 4, 5]:
            try:
                textrank_sentences = self.textrank_summary(text, sent_count)

                # 创建引导式prompt
                guided_prompt = f"""
                Key sentences extracted from the document: {textrank_sentences}
                
                Original document: {text[:800]}...
                
                Task: Based on the key sentences above, generate a coherent and comprehensive summary that covers all the important information while maintaining natural flow.
                
                Summary:"""

                # 使用BART生成基于引导的摘要
                bart_guided = self.bart_summary(
                    guided_prompt, max_length=120, min_length=40)
                candidates.append({
                    'summary': bart_guided,
                    'type': f'guided_{sent_count}_sentences',
                    'extracted_sentences': textrank_sentences
                })

            except Exception as e:
                print(
                    f"Error generating candidate with {sent_count} sentences: {e}")
                continue

        # 添加重写式候选
        try:
            textrank_base = self.textrank_summary(text, 4)
            rewrite_prompt = f"""
            Extracted summary: {textrank_base}
            
            Task: Rewrite the above extracted summary to make it more coherent and fluent while preserving all the core information and coverage.
            
            Rewritten summary:"""

            rewritten = self.bart_summary(
                rewrite_prompt, max_length=130, min_length=50)
            candidates.append({
                'summary': rewritten,
                'type': 'rewritten',
                'extracted_sentences': textrank_base
            })
        except Exception as e:
            print(f"Error generating rewritten candidate: {e}")

        return candidates

    def evaluate_summary(self, summary, reference):
        """评估摘要的ROUGE和BERTScore"""
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, summary)
        rouge_dict = {
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure
        }

        # BERTScore
        try:
            P, R, F1 = bert_score([summary], [reference],
                                  lang='en', verbose=False)
            bert_dict = {
                'bertscore_precision': P.item(),
                'bertscore_recall': R.item(),
                'bertscore_f1': F1.item()
            }
        except Exception as e:
            print(f"BERTScore error: {e}")
            bert_dict = {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

        return {**rouge_dict, **bert_dict}

    def weighted_fusion(self, candidates, reference):
        """加权融合选择最佳候选"""
        best_candidate = None
        best_score = -1
        candidate_scores = []

        for candidate in candidates:
            scores = self.evaluate_summary(candidate['summary'], reference)

            # 加权评分 (ROUGE + BERTScore)
            weighted_score = (
                scores['rouge1_f'] * 0.25 +
                scores['rouge2_f'] * 0.25 +
                scores['rougeL_f'] * 0.25 +
                scores['bertscore_f1'] * 0.25
            )

            candidate_scores.append({
                'candidate': candidate,
                'scores': scores,
                'weighted_score': weighted_score
            })

            if weighted_score > best_score:
                best_score = weighted_score
                best_candidate = candidate

        return best_candidate, candidate_scores

    def compare_methods(self, text, reference_summary=None):
        """对比三种方法的效果"""
        if reference_summary is None:
            reference_summary = text  # 使用原文作为参考

        print("="*80)
        print("SUMMARY COMPARISON EVALUATION")
        print("="*80)

        # 1. TextRank摘要
        print("\n1. Generating TextRank summary...")
        textrank_summary = self.textrank_summary(text, 5)
        textrank_scores = self.evaluate_summary(
            textrank_summary, reference_summary)

        print(f"TextRank Summary:\n{textrank_summary}\n")

        # 2. BART摘要
        print("2. Generating BART summary...")
        bart_summary = self.bart_summary(text)
        bart_scores = self.evaluate_summary(bart_summary, reference_summary)

        print(f"BART Summary:\n{bart_summary}\n")

        # 3. 混合摘要
        print("3. Generating hybrid summary candidates...")
        candidates = self.generate_hybrid_candidates(text)

        if not candidates:
            print("No valid candidates generated!")
            return

        best_hybrid, all_candidates = self.weighted_fusion(
            candidates, reference_summary)
        hybrid_scores = self.evaluate_summary(
            best_hybrid['summary'], reference_summary)

        print(
            f"Best Hybrid Summary ({best_hybrid['type']}):\n{best_hybrid['summary']}\n")

        # 结果对比表格
        print("="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        methods = ['TextRank', 'BART', 'Hybrid']
        scores_list = [textrank_scores, bart_scores, hybrid_scores]

        # 打印表头
        print(
            f"{'Method':<12} {'ROUGE-1':<8} {'ROUGE-2':<8} {'ROUGE-L':<8} {'BERTScore':<10}")
        print("-" * 50)

        # 打印各方法得分
        for method, scores in zip(methods, scores_list):
            print(f"{method:<12} {scores['rouge1_f']:<8.4f} {scores['rouge2_f']:<8.4f} "
                  f"{scores['rougeL_f']:<8.4f} {scores['bertscore_f1']:<10.4f}")

        # 计算提升百分比
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)

        def calculate_improvement(base_scores, target_scores):
            improvements = {}
            for key in base_scores:
                if base_scores[key] > 0:
                    improvement = (
                        (target_scores[key] - base_scores[key]) / base_scores[key]) * 100
                    improvements[key] = improvement
                else:
                    improvements[key] = 0
            return improvements

        # 相比TextRank的提升
        textrank_improvement = calculate_improvement(
            textrank_scores, hybrid_scores)
        print(f"\nHybrid vs TextRank:")
        print(f"  ROUGE-1: {textrank_improvement['rouge1_f']:+.2f}%")
        print(f"  ROUGE-2: {textrank_improvement['rouge2_f']:+.2f}%")
        print(f"  ROUGE-L: {textrank_improvement['rougeL_f']:+.2f}%")
        print(f"  BERTScore: {textrank_improvement['bertscore_f1']:+.2f}%")

        # 相比BART的提升
        bart_improvement = calculate_improvement(bart_scores, hybrid_scores)
        print(f"\nHybrid vs BART:")
        print(f"  ROUGE-1: {bart_improvement['rouge1_f']:+.2f}%")
        print(f"  ROUGE-2: {bart_improvement['rouge2_f']:+.2f}%")
        print(f"  ROUGE-L: {bart_improvement['rougeL_f']:+.2f}%")
        print(f"  BERTScore: {bart_improvement['bertscore_f1']:+.2f}%")

        # 显示所有候选摘要的得分
        print(f"\n" + "="*80)
        print("ALL HYBRID CANDIDATES PERFORMANCE")
        print("="*80)

        for i, candidate_info in enumerate(all_candidates):
            candidate = candidate_info['candidate']
            scores = candidate_info['scores']
            weighted = candidate_info['weighted_score']

            print(f"\nCandidate {i+1} ({candidate['type']}):")
            print(f"  Summary: {candidate['summary'][:100]}...")
            print(f"  Weighted Score: {weighted:.4f}")
            print(f"  ROUGE-1: {scores['rouge1_f']:.4f}, ROUGE-2: {scores['rouge2_f']:.4f}, "
                  f"ROUGE-L: {scores['rougeL_f']:.4f}, BERTScore: {scores['bertscore_f1']:.4f}")

        return {
            'textrank': {'summary': textrank_summary, 'scores': textrank_scores},
            'bart': {'summary': bart_summary, 'scores': bart_scores},
            'hybrid': {'summary': best_hybrid['summary'], 'scores': hybrid_scores},
            'improvements': {
                'vs_textrank': textrank_improvement,
                'vs_bart': bart_improvement
            }
        }

# 使用示例


def main():
    # 示例文档
    sample_text = """
    Artificial intelligence (AI) has become one of the most transformative technologies of the 21st century, 
    revolutionizing industries from healthcare to finance. Machine learning, a subset of AI, enables computers 
    to learn and improve from experience without being explicitly programmed. Deep learning, which uses neural 
    networks with multiple layers, has achieved remarkable breakthroughs in image recognition, natural language 
    processing, and game playing. Companies like Google, Microsoft, and OpenAI are investing billions of dollars 
    in AI research and development. However, the rapid advancement of AI also raises important ethical concerns 
    about job displacement, privacy, and the potential for bias in algorithmic decision-making. Governments 
    worldwide are grappling with how to regulate AI while still fostering innovation. The future of AI promises 
    even greater integration into daily life, with applications in autonomous vehicles, smart cities, and 
    personalized medicine. As AI continues to evolve, it will be crucial to ensure that its benefits are 
    distributed equitably and that its risks are carefully managed.
    """

    # 初始化评估器
    evaluator = HybridSummaryEvaluator()

    # 运行对比评估
    results = evaluator.compare_methods(sample_text)

    print(f"\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    # 安装依赖提示
    print("请确保已安装以下依赖包:")
    print("pip install rouge-score bert-score transformers sumy torch")
    print("python -m spacy download en_core_web_sm")
    print("\n开始评估...\n")

    main()
