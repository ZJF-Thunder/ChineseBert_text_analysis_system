import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader

# 加载数据集
df = pd.read_csv('THUCNews.csv', sep='\t', header=None, names=['label', 'text'])
# 标签映射
label_map = { '财经': 0, '彩票': 1, '房产': 2, '股票': 3, '家居': 4, '教育': 5, '科技': 6, '社会': 7, '时尚': 8, '时政': 9, '游戏': 10, '娱乐': 11 }
df['label'] = df['label'].map(label_map)
# 分割训练集和测试集
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)


# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=12)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义训练函数
def train_model(train_df, model, tokenizer):
    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()
    # 数据加载器
    train_loader = DataLoader(train_df, batch_size=32, shuffle=True)
    # 训练循环
    model.train()
    for epoch in range(5):
        for batch in train_loader:
            # 编码文本
            input_ids = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').input_ids
            # 计算损失
            outputs = model(input_ids, labels=batch['label'])
            loss = criterion(outputs.logits, batch['label'])
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch} done.')
    return model

# 训练模型
trained_model = train_model(train_df, model, tokenizer)


def predict(text, model, tokenizer):
    # 编码文本
    input_ids = tokenizer(text, padding=True, truncation=True, return_tensors='pt').input_ids
    # 预测类别
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, dim=1)
    return predicted.item()

# 测试预测函数
text = '这个手机真的很好用'
predicted_label = predict(text, trained_model, tokenizer)
print(f'Predicted label: {predicted_label}')
