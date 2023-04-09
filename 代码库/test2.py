# 本地文件夹加载chinese-bert-wwm-ext模型
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext", num_classes=2)
model = BertForSequenceClassification.from_pretrained("chinese-bert-wwm-ext")
# Hugging Face API 调用
# encoded_input = tokenizer("这是一条文本", padding=True, truncation=True, return_tensors='pt')
# output = model(**encoded_input)

# 本地调用
input_ids = tokenizer.encode("今天中国进了世界杯", padding=True, truncation=True, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
output = model(input_ids=input_ids, attention_mask=attention_mask)
# 获取分类结果
logits = output.logits
print("分类结果：", logits)

label_map = model.config.id2label
predicted_label = label_map[logits.argmax().item()]
print("分类结果：", predicted_label)

# result = logits.detach().numpy()[0]
# pred = torch.argmax(logits, dim=1)
# label_name = label_list[pred]
# print("预测标签为：", label_name)
#
# print("分类结果：", result)