from transformers import BartTokenizer, BartForConditionalGeneration, LogitsProcessor, LogitsProcessorList
import evaluate
import torch

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Input document (source) and reference summary (gold standard)
input_text = """
Climate change is causing more frequent and severe weather events around the world, 
including hurricanes, droughts, and wildfires. Scientists are calling for immediate action 
to reduce carbon emissions and transition to renewable energy sources.
"""

reference_summary = """
Scientists call for urgent action to reduce carbon emissions and promote renewable energy, 
addressing the worsening impact of climate change.
"""

# Define monitor pool (important phrases we want to ensure appear in the summary)
monitor_pool = {
    "climate change": False,
    "carbon emissions": False,
    "renewable energy": False
}

# Custom logits processor to boost importance of missing keywords


class DynamicContentLogitsProcessor(LogitsProcessor):
    def __init__(self, monitor_pool, tokenizer):
        self.monitor_pool = monitor_pool
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        generated_text = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True).lower()

        # Update monitor pool: mark keywords as covered
        for keyword in self.monitor_pool:
            if not self.monitor_pool[keyword] and keyword in generated_text:
                self.monitor_pool[keyword] = True

        # Boost logits for uncovered keywords
        for keyword, covered in self.monitor_pool.items():
            if not covered:
                token_ids = self.tokenizer.encode(
                    keyword, add_special_tokens=False)
                for token_id in token_ids:
                    if token_id < scores.size(-1):
                        # Increase likelihood of generating token
                        scores[:, token_id] += 2.0

        return scores


# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt",
                   max_length=1024, truncation=True)

# Setup dynamic logits processor
logits_processor = LogitsProcessorList([
    DynamicContentLogitsProcessor(monitor_pool, tokenizer)
])

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        num_beams=4,
        early_stopping=True,
        logits_processor=logits_processor
    )

# Decode generated token IDs
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Evaluate summary using ROUGE
rouge = evaluate.load("rouge")
scores = rouge.compute(
    predictions=[generated_summary], references=[reference_summary])

# Print results
print("\n=== Final Dynamic-Controlled Summary ===")
print(generated_summary)

print("\n=== Monitor Pool Status ===")
for keyword, covered in monitor_pool.items():
    print(f"{keyword}: {'Covered' if covered else 'Not Covered'}")

print("\n=== ROUGE Evaluation Scores ===")
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")
print(f"ROUGE-L: {scores['rougeL']:.4f}")
