# 临时参考代码   给出的优化代码
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split

# 载入数据集
df = pd.read_csv('THUCNews.csv')
df = df[['label', 'text']]

# 对标签进行映射
label_map = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9, '社会': 10,
             '股票': 11}
df['label'] = df['label'].map(label_map)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 加载Bert预训练模型和分词器
model_name = 'chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=12)

# 设置训练超参数
batch_size = 8
epochs = 3
learning_rate = 5e-5


# 将数据转化为Bert需要的格式
def convert_data_to_feature(df):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for text, label in zip(df['text'], df['label']):
        encoded_dict = tokenizer.encode_plus(text,
                                             add_special_tokens=True,
                                             max_length=128,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    return dataset


# 转换训练数据和测试数据
train_dataset = convert_data_to_feature(train_df)
test_dataset = convert_data_to_feature(test_df)

# 创建随机采样器和顺序采样器
train_sampler = RandomSampler(train_dataset)
test_sampler = SequentialSampler(test_dataset)

# 创建数据加载器
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 开始训练
