import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('../models/bert-base-chinese')
model = BertModel.from_pretrained('../models/bert-base-chinese')

sentence = "The quick brown fox jumps over the lazy dog"
new_word = "cat"

# 将原始句子转换成 BERT 模型输入格式
inputs = tokenizer(sentence, return_tensors='pt', padding=True)

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取句子中每个单词的向量表示
embeddings = outputs.last_hidden_state

# 获取新单词的向量表示
new_word_id = tokenizer.convert_tokens_to_ids(new_word)
new_word_embedding = model.embeddings.word_embeddings.weight[new_word_id]
# new_word_embedding = model.bert.embeddings.word_embeddings.weight[new_word_id]
# new_word_embedding = model.transformer.wte.weight[new_word_id]

# 在原始句子的第 3 个位置插入新单词的向量表示
insert_pos = 2
masked_embeddings = embeddings.clone()
masked_embeddings[0, insert_pos] = new_word_embedding

# 打印结果
print(masked_embeddings)
