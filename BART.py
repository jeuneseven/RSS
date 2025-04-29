from transformers import BartTokenizer, BartForConditionalGeneration

# 选择模型名字
model_name_cnn = "facebook/bart-large-cnn"
model_name_xsum = "facebook/bart-large-xsum"

# 加载CNN模型
tokenizer_cnn = BartTokenizer.from_pretrained(model_name_cnn)
model_cnn = BartForConditionalGeneration.from_pretrained(model_name_cnn)

# 加载XSUM模型
tokenizer_xsum = BartTokenizer.from_pretrained(model_name_xsum)
model_xsum = BartForConditionalGeneration.from_pretrained(model_name_xsum)

# 示例输入文本（可以换成你自己的文本）
input_text = """
Climate change is causing more frequent and severe weather events around the world, 
including hurricanes, droughts, and wildfires. Scientists are calling for immediate action 
to reduce carbon emissions and transition to renewable energy sources.
"""

# 统一封装一个推理函数


def summarize(text, model, tokenizer, max_input_length=1024, max_output_length=150):
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=max_input_length, truncation=True)
    # 生成摘要
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    # 解码输出
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# 用CNN模型生成摘要
summary_cnn = summarize(input_text, model_cnn, tokenizer_cnn)
print("\n=== Summary using BART-large-CNN ===")
print(summary_cnn)

# 用XSUM模型生成摘要
summary_xsum = summarize(input_text, model_xsum, tokenizer_xsum)
print("\n=== Summary using BART-large-XSUM ===")
print(summary_xsum)
