# 数据集预处理
# 解压原始数据集，将Rumor_Dataset.zip解压至data目录下
import json
import os
import random
import zipfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

# import paddle
# import paddle.fluid as fluid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available')

src_path = "./Rumor_Dataset.zip"
target_path = "./Chinese_Rumor_Dataset-master"
if not os.path.isdir(target_path):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()

rumor_class_dirs = os.listdir(target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
non_rumor_class_dirs = os.listdir(target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
original_microblog = target_path + "/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"

# 谣言为0，非谣言为1
rumor_label = "0"
non_rumor_label = "1"

rumor_num = 0
non_rumor_num = 0

all_rumor_list = []
all_non_rumor_list = []

# 解析谣言数据
for rumor_class_dir in rumor_class_dirs:
    if rumor_class_dir != '.DS_Store':
        with open(original_microblog + rumor_class_dir, 'r', encoding='utf-8') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1

# 解析非谣言数据
for non_rumor_class_dir in non_rumor_class_dirs:
    if non_rumor_class_dir != '.DS_Store':
        with open(original_microblog + non_rumor_class_dir, 'r', encoding='utf-8') as f2:
            non_rumor_content = f2.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1

print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))

data_list_path = "./data/"
all_data_path = data_list_path + "all_data.txt"
# 将谣言列表和非谣言列表连接成一个新的列表
all_data_list = all_rumor_list + all_non_rumor_list
# 将文本打乱顺序
random.shuffle(all_data_list)

with open(all_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a', encoding='utf-8') as f:
    for data in all_data_list:
        f.write(data)


# 生成数据字典
def create_dict(data_path, dict_path):
    """
    :param data_path: 样本数据路径
    :param dict_path: 生成字典路径
    :return: None
    """
    dict_set = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 用列表推导式遍历dict_list列表中的含有字和字的序号的每一个子列表，获取key和value
    # key为每个子列表的第一个元素，value为每个子列表的第二个元素
    keys = [sublist[0] for sublist in dict_list]
    values = [sublist[1] for sublist in dict_list]
    dict_txt = dict(zip(keys, values))
    # dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成成功！")


# 获取字典的长度
def get_dict_len(dict_path):
    """
    :param dict_path: 字典的路径
    :return: 字典长度
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())


# 按8：1划分训练集和数据集
def create_data_list(data_list_path):
    """
    :param data_list_path: 样本数字化文本的指定生成路径
    :return: 谣言和非谣言的数字化文本  文本向量化
    """
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'w', encoding='utf-8') as f_eval:
        f_eval.seek(0)
        f_eval.truncate()

    with open(os.path.join(data_list_path, 'train_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()

    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, \
            open(os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            words = line.split('\t')[-1].replace('\n', '')
            label = line.split('\t')[0]
            labs = ""
            if i % 8 == 0:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_eval.write(labs)
            else:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_train.write(labs)
            i += 1

    print("数据列表生成成功！")


# 数据字典
dict_path = data_list_path + "dict.txt"

with open(dict_path, 'w') as f:
    f.seek(0)
    f.truncate()

create_dict(all_data_path, dict_path)
create_data_list(data_list_path)

"""
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                data = [int(x) for x in line[0].split(',')]
                label = int(line[1])
                self.samples.append((torch.tensor(data), torch.tensor(label)))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

batch_size = 16
train_dataset = MyDataset('train.txt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = Chinesebert.from_pretrained('uer/chinese_roberta_L-8_H-768_A-12', num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):
    for i, batch in enumerate(train_loader):
        data, label = batch
        optimizer.zero_grad()
        loss, _ = model(data, labels=label)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
"""

# 用BertTokenizer来作为模型加载

# import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', num_classes=2)
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
model = BertForSequenceClassification.from_pretrained('chinese-bert-wwm-ext')


# 定义一个函数来读取数据并将其转换为模型所需的格式
def read_data(filename):
    """
    :param filename: 需要转换的数据的路径
    :return: 转换成模型所需的格式的数据
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        text, label = line.strip().split('\t')
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        data.append((inputs.input_ids, inputs.token_type_ids, inputs.attention_mask, int(label)))

    return data



def read_data2(filename):
    """
    :param filename: 需要转换的数据的路径
    :return: 转换成模型所需的格式的数据
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        label, text = line.strip().split('\t')
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        data.append((inputs.input_ids, inputs.token_type_ids, inputs.attention_mask, int(label)))

    return data


train_path = './data/train_list2.txt'
# train_path = './data/train_list.txt'
eval_path = './data/eval_list.txt'
# 读取训练集和测试集
data = read_data2('./data/all_data.txt')
train_data = read_data(train_path)   # list:2963 ,里面是一个包含了4个元素的元组，四个元素中，有三个是张量，还有一个是整数
test_data = read_data(eval_path)

# 将数据转换为PyTorch张量

train_tensor = [
    (torch.tensor(input_ids).cuda(), torch.tensor(token_type_ids).cuda(), torch.tensor(attention_mask).cuda(), torch.tensor([[label]]).cuda()) for
    input_ids, token_type_ids, attention_mask, label in train_data]
# train_tensor = tuple(torch.tensor(item).cuda() for item in train_tensor)

# train_tensor = torch.tensor([item.cpu().detach().numpy() for item in train_tensor]).cuda()


# train_inputs = [(torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)) for
#                 input_ids, token_type_ids, attention_mask, label in train_data]
# train_labels = torch.tensor([label for input_ids, token_type_ids, attention_mask, label in train_data])


test_inputs = [(torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)) for
               input_ids, token_type_ids, attention_mask, label in test_data]
# test_inputs = torch.tensor([item.cpu().detach().numpy() for item in test_inputs]).cuda()
# for item in train_tensor:
#     print("第一个张量的维度：", item.shape)

test_labels = torch.tensor([label for input_ids, token_type_ids, attention_mask, label in test_data])

# 将数据加载到模型中进行微调
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
# train_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
# train_dataset = torch.utils.data.TensorDataset(*zip(*train_inputs), train_labels)
# train_dataset = torch.utils.data.TensorDataset(*{'inputs': train_inputs, 'labels': train_labels}.values())

# 列表

# train_data2 = [(train_inputs[i], train_labels[i]) for i in range(len(train_inputs))]
# train_tensor = [torch.tensor(data) for data in train_data2]
# 使用 torch.is_tensor() 检查是否是张量
# print("是否为张量：", torch.is_tensor(torch.Tensor(train_tensor)))  # True

# 使用 dtype 属性查看数据类型
# train_tensor = torch.tensor(train_tensor)
# print("是否为整数：", train_tensor.dtype)

# print("是否为整数：", train_tensor.dtype) # torch.int32

# train_tensor_stacked = tuple(torch.stack(tensors) for tensors in train_tensor)
# train_dataset = TensorDataset(*train_tensor_stacked)
# train_dataset = TensorDataset(*train_tensor)
train_dataset = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_tensor)])


# train_tensor = torch.tensor(train_data2)
# train_dataset = TensorDataset(train_tensor)


# train_dataset = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, drop_last=True)
model.train()
# 将模型移动到GPU设备
model = model.to('cuda')
# 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available')

for epoch in range(3):
    i = 0
    for input_ids, token_type_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.squeeze()  # 二维变成一维
        token_type_ids = token_type_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        labels = labels.squeeze()
        # 将模型的参数梯度清零，以免上一个epoch的梯度对当前epoch的训练产生影响
        optimizer.zero_grad()
        # 将模型移动到GPU设备
        # model = model.cuda()
        # model = model.to('cuda')
        print("input_ids的形状：", input_ids.size())
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        i += 1
        print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
        print(outputs)
        print("input_ids的形状：", input_ids.size())
        # if i % 10 == 0:
        #     print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))

# 在测试集上进行评估
test_dataset = torch.utils.data.TensorDataset(test_inputs, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for input_ids, token_type_ids, attention_mask, labels in test_loader:
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits)
        correct += (predictions == labels).sum().item()
        total += labels.shape[0]
accuracy = correct / total
print('Test accuracy:', accuracy)




import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

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




def data_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)


"""
def data_reader(data_path):
    def reader():
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


BATCH_SIZE = 128

train_list_path = data_list_path + 'train_list.txt'
eval_list_path = data_list_path + 'eval_list.txt'

train_reader = paddle.batch(
    reader=data_reader(train_list_path),
    batch_size=BATCH_SIZE)
eval_reader = paddle.batch(
    reader=data_reader(eval_list_path),
    batch_size=BATCH_SIZE)

"""
