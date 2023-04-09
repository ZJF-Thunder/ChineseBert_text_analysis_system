from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 加载chinesebert的tokenizer和预训练模型
tokenizer = AutoTokenizer.from_pretrained("chinese-bert-wwm-ext", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("chinese-bert-wwm-ext", num_labels=2, local_files_only=True)

# 用tokenizer对文本进行编码
inputs = tokenizer("这是一条文本", return_tensors="pt")

# 使用预训练模型进行推理
outputs = model(**inputs)

# 获取分类结果
logits = outputs.logits
print("分类结果：", logits)
result = logits.detach().numpy()[0]
print("分类结果：", result)
