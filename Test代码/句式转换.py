# 定义一个BERT模型并使用其进行词汇插入
import torch
from transformers import BertTokenizer, BertModel

# 加载BERT模型及其tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('../models/bert-base-chinese')
model = BertModel.from_pretrained('../models/bert-base-chinese')
# 定义原始句子和需要插入的单词
text = "The quick brown fox jumps over the lazy dog"
new_word = "smart"

# 对句子进行tokenize，并在需要插入单词的位置插入[MASK]符号
tokenized_text = tokenizer.tokenize(text)
mask_index = tokenized_text.index("brown")
tokenized_text[mask_index] = "[MASK]"

# 将token转换为id，并将其放入tensor中
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# 在模型中运行输入的句子
with torch.no_grad():
    outputs = model(tokens_tensor)

# 获取需要插入单词的位置的向量表示
masked_embeddings = outputs[0][0][mask_index]

# 计算新单词的向量表示并插入到原始句子中
# new_word_embedding = model.embeddings.word_embeddings.weight[new_word_id]
new_word_embeddings = masked_embeddings + model.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(new_word)]
tokenized_text[mask_index] = new_word
final_text = tokenizer.convert_tokens_to_string(tokenized_text)
print(final_text)

# 在上述代码中，我们首先加载BERT模型及其tokenizer，在原始句子中找到需要替换的单词，并将其替换为[MASK]。
# 然后，我们将标记的文本输入BERT模型中，并获取需要替换的单词所在位置的向量表示。
# 最后，我们从词汇表中获取要插入的新单词对应的向量表示，并将其与原始向量加和得到新的向量表示。
# 最终，我们可以将原始句子中的[MASK]替换成新的单词。
# 接下来，我们可以使用掩码语言模型任务来进行句式转化：


# 微调BERT模型以处理MLM任务
model.train()

# 定义原始句子
text = "The quick brown fox jumps over the lazy dog"

# 对句子进行tokenize，并在随机选择的位置插入[MASK]符号
tokenized_text = tokenizer.tokenize(text)
mask_index = 3
tokenized_text[mask_index] = "[MASK]"

# 将token转换为id，并将其放入tensor中
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# 运行模型以获得mask处的隐藏表示
with torch.no_grad():
    outputs = model(tokens_tensor)

# 获取mask处的隐藏表示
masked_embeddings = outputs[0][0][mask_index]
print(masked_embeddings)
# 使用隐藏表示预测新单词
softmax_weights = model.embeddings.word_embeddings.weight
print(softmax_weights)
# softmax_weights = model.cls.predictions.decoder.weight
new_tokens_logits = torch.matmul(masked_embeddings, softmax_weights.T)
new_tokens_probs = torch.softmax(new_tokens_logits, dim=-1)
top_new_tokens = torch.topk(new_tokens_probs, k=5).indices[0].tolist()
print(top_new_tokens)
# 替换mask处的标记为预测到的单词
for new_token in top_new_tokens:
    predicted_token = tokenizer.convert_ids_to_tokens([new_token])[0]
    tokenized_text[mask_index] = predicted_token
    print(tokenizer.convert_tokens_to_string(tokenized_text))