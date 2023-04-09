# 文本分类和情感分析系统
import pandas as pd

# 加载数据集
df = pd.read_csv('THUCNews.csv', sep='\t', header=None, names=['label', 'text'])
# 标签映射
label_map = { '财经': 0, '彩票': 1, '房产': 2, '股票': 3, '家居': 4, '教育': 5, '科技': 6, '社会': 7, '时尚': 8, '时政': 9, '游戏': 10, '娱乐': 11 }
df['label'] = df['label'].map(label_map)
# 分割训练集和测试集
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

