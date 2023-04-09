"""
功能：加载微博谣言数据集，微调预训练模型并以此进行判断是否为谣言
"""
import json
import os
import random
import zipfile

import matplotlib
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, BertConfig
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
from IPython.display import clear_output
import logging


# 数据预处理：输入数据，并将数据处理成模型可接受的形式
# 解压原始数据集，将Rumor_Dataset.zip解压至data目录下
def data_preprocessing(src_path, target_path):
    """
    :param src_path: 需要解压的文件的路径
    :param target_path: 解压之后的文件存放路径
    :return: 随机打乱的数据列表
    """
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
    logging.info("谣言数据总量为：" + str(rumor_num))
    logging.info("非谣言数据总量为：" + str(non_rumor_num))
    # 将谣言列表和非谣言列表连接成一个新的列表
    all_data_list = all_rumor_list + all_non_rumor_list
    # 将文本打乱顺序
    random.shuffle(all_data_list)
    # 返回随机打乱的数据列表
    return all_data_list


# WordPiece分词方法生成数据字典
def create_dict(data_path, data_list, dict_path):
    """
    :param data_path: 样本数据路径
    :param dict_path: 生成字典路径
    :return: None
    """
    # 清空文本的数据，并往all_data_path的txt文本中写入数据
    with open(data_path, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(data_path, 'a', encoding='utf-8') as f:
        for data in data_list:
            f.write(data)
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
    # 清空文本
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成成功！")
    logging.info("数据字典生成成功！")


# 获取字典的长度
def get_dict_len(dict_path):
    """
    :param dict_path: 字典的路径
    :return: 字典长度
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())


# 按照9：1划分数据集
def splitting_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    with open('./data/train_data.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open('./data/test_data.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_data)


# 按8：1划分训练集和数据集,生成数据·对应的数据集txt向量化文本
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
    logging.info("数据列表生成成功！")


# 定义一个函数来读取数据并将其转换为模型所需的格式
def read_data(filename, tokenizer):
    """
    :param filename: 需要转换的数据的路径
    :return: 转换成模型所需的格式的数据
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        # text, label = line.strip().split('\t')
        label, text = line.strip().split('\t')
        # 样例
        # encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, label=label)
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # add_special_tokens = True将特殊标记如[CLS]和[SEP]添加到输入序列的开头和结尾。
        # padding = 'max_length'  将序列填充到最大长度。
        # truncation = True 如果序列超过最大长度，则将其截断到最大长度。
        # max_length = 32 指定编码序列的最大长度。
        # return_attention_mask = True返回一个注意力掩码，指示哪些标记是填充标记。
        # return_token_type_ids = True返回标记类型ID，指示哪些标记属于哪个段落（对于问答等任务很有用）。
        # return_tensors = 'pt' 返回PyTorch张量而不是列表。
        data.append((inputs.input_ids, inputs.token_type_ids, inputs.attention_mask, int(label)))
    # 将数据转换为PyTorch张量,并在GPU上计算
    # 改进版本:下面两种方式都可以,不会警告
    # data_tensor = [
    #     (torch.as_tensor(input_ids).cuda(), torch.as_tensor(token_type_ids).cuda(), torch.as_tensor(attention_mask).cuda(),
    #      torch.tensor([[label]]).cuda()) for input_ids, token_type_ids, attention_mask, label in data]
    data_tensor = [
        (input_ids.clone().clone().detach().cuda(), token_type_ids.clone().clone().detach().cuda(),
         attention_mask.clone().clone().detach().cuda(),
         torch.tensor([[label]]).cuda()) for input_ids, token_type_ids, attention_mask, label in data]
    return data_tensor


# 定义一个函数来读取数据并将其转换为模型所需的格式
def read_data2(filename, tokenizer):
    """
    :param filename: 需要转换的数据的路径
    :return: 转换成模型所需的格式的数据
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        text, label = line.strip().split('\t')
        # label,  text= line.strip().split('\t')
        # 样例
        # encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, label=label)
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # add_special_tokens = True将特殊标记如[CLS]和[SEP]添加到输入序列的开头和结尾。
        # padding = 'max_length'  将序列填充到最大长度。
        # truncation = True 如果序列超过最大长度，则将其截断到最大长度。
        # max_length = 32 指定编码序列的最大长度。
        # return_attention_mask = True返回一个注意力掩码，指示哪些标记是填充标记。
        # return_token_type_ids = True返回标记类型ID，指示哪些标记属于哪个段落（对于问答等任务很有用）。
        # return_tensors = 'pt' 返回PyTorch张量而不是列表。
        data.append((inputs.input_ids, inputs.token_type_ids, inputs.attention_mask, int(label)))
    # 将数据转换为PyTorch张量,并在GPU上计算
    # 改进版本:下面两种方式都可以,不会警告
    # data_tensor = [
    #     (torch.as_tensor(input_ids).cuda(), torch.as_tensor(token_type_ids).cuda(), torch.as_tensor(attention_mask).cuda(),
    #      torch.tensor([[label]]).cuda()) for input_ids, token_type_ids, attention_mask, label in data]
    data_tensor = [
        (input_ids.clone().clone().detach().cuda(), token_type_ids.clone().clone().detach().cuda(),
         attention_mask.clone().clone().detach().cuda(),
         torch.tensor([[label]]).cuda()) for input_ids, token_type_ids, attention_mask, label in data]
    return data_tensor


# 新增部分，增加对标签的独热编码
def one_hot_encoding(label_ids, num_labels):
    labels = torch.zeros((len(label_ids), num_labels))
    labels[torch.arange(len(label_ids)), label_ids] = 1
    return labels


# 训练模型
def model_train(model, train_tensor):
    train_tensor_stack = [torch.stack(tensors) for tensors in zip(*train_tensor)]
    train_dataset = TensorDataset(*train_tensor_stack)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, drop_last=True)

    # 将数据加载到模型中进行微调
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    # 增加学习率调整机制
    total_steps = len(train_loader) * Epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # 将模型移动到GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    start_time = time.time()

    train_losses = []
    iters = 0
    # fig, ax = plt.subplots()
    # ax.set(xlim=(0, 380), ylim=(0, 1))
    # def animate(i):
    #     ax.clear()
    #     ax.plot(range(i + 1), train_losses[:i + 1], label='Training Loss')
    #     ax.legend(loc='upper right')
    #     ax.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
    #     fig.canvas.draw()
    #     plt.pause(0.1)
    #     return ax.lines
    #     # plt.pause(0.1)
    #     # return ax.lines  # 显式地返回绘制的 artist 对象
    #
    # ani = FuncAnimation(fig, animate, frames=380, repeat=True, blit=True)
    for epoch in range(Epochs):

        epoch_loss = 0  # 每一个Epoch的平均loss
        # 1
        # for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
        #     # 正向传播
        #     input_ids = input_ids.squeeze(1).to(device)
        #     token_type_ids = token_type_ids.squeeze(1).to(device)
        #     attention_mask = attention_mask.squeeze(1).to(device)
        #     labels = labels.to(device)

        # 2
        for i, data in enumerate(train_loader):
            # data = (input_ids, token_type_ids, attention_mask, labels)
            # 正向传播
            # input_ids = data[0].squeeze().to(device)
            # token_type_ids = data[1].squeeze().to(device)
            # attention_mask = data[2].squeeze().to(device)
            # labels = data[3].squeeze().to(device)

            input_ids = data[0].to(device).squeeze(1)
            token_type_ids = data[1].to(device).squeeze(1)
            attention_mask = data[2].to(device).squeeze(1)
            labels = data[3].to(device).squeeze()
            # 3
            # for input_ids, token_type_ids, attention_mask, labels in train_loader:
            #     # 正向传播
            #     input_ids = input_ids.squeeze()  # 二维变成一维
            #     token_type_ids = token_type_ids.squeeze()
            #     attention_mask = attention_mask.squeeze()
            #     labels = labels.squeeze()
            # 将模型的参数梯度清零，以免上一个epoch的梯度对当前epoch的训练产生影响
            optimizer.zero_grad()
            # 新增部分：将标签进行独热编码，然后输入到模型中，并且调用二分类交叉损失函数
            labels = one_hot_encoding(labels, 2)
            labels = labels.cuda()
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

            # 这是模型自带的损失函数,直接loss = outputs.loss调用即可
            # loss = outputs.loss
            # labels = torch.zeros((Batch_size, 2))
            # labels[torch.arange(Batch_size), labels] = 1

            # 调用交叉熵损失函数计算损失,labels及性能独热编码之后,本身就是浮点数,所以加不加float都一样
            loss = criterion(outputs.logits, labels.float())
            # loss = criterion(outputs.logits.flatten(), labels.squeeze())
            # loss = criterion(outputs.logits.flatten(), labels.squeeze())
            # loss = criterion(outputs.squeeze(), labels.float())
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 调整学习率
            scheduler.step()
            # i += 1
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            iters += 1
            if i % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
                logging.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
        avg_epoch_loss = epoch_loss / len(train_loader)
        print("Epoch [{}/{}], Average Loss: {:.4f}".format(epoch + 1, Epochs, avg_epoch_loss))
        logging.info("Epoch [{}/{}], Average Loss: {:.4f}".format(epoch + 1, Epochs, avg_epoch_loss))

    end_time = time.time()
    total_time = end_time - start_time
    print("————————————————模型训练完成————————————————")
    print("训练开始时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print("训练结束时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print(f"模型训练总时间为: {total_time:.2f} 秒")
    logging.info("————————————————模型保存成功————————————————")
    logging.info("训练开始时间：" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    logging.info("训练结束时间：" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    logging.info(f"模型训练总时间为: {total_time:.2f} 秒")
    # 获取当前时间
    local_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(end_time))
    save_dir = './模型保存'  # 保存模型的相对路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果目录不存在，创建目录
    model_name = os.path.join(save_dir, "ChineseBert_{}.pt".format(local_time))  # 模型的相对路径
    torch.save(model.state_dict(), model_name)
    print("————————————————模型保存成功————————————————")
    logging.info("————————————————模型保存成功————————————————")

    # 绘制动态图
    # 清除控制台的输出
    # clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    # plt.show()
    img_save_dir = './images'
    images = os.path.join(img_save_dir, "training_loss_{}.png".format(local_time))
    plt.savefig(images)
    fig, ax = plt.subplots()
    ax.set(xlim=(0, iters), ylim=(0, max(train_losses)))

    def animate(i):
        ax.clear()
        ax.plot(range(i + 1), train_losses[:i + 1], label='Training Loss')
        ax.legend(loc='upper right')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
        plt.pause(0.1)
        # plt.savefig('figure.png')

    ani = FuncAnimation(fig, animate, frames=iters, repeat=True)
    # ani.save('animation.gif', writer='pillow')
    # plt.show()
    # ani.save('animation2.gif', writer='pillow')
    # ani.save('animation2.jpg')
    return model


def show_animation(train_losses, fig, ax):
    def animate(i):
        ax.clear()
        ax.plot(range(i + 1), train_losses[:i + 1], label='Training Loss')
        ax.legend(loc='upper right')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
        return ax.lines

    ani = FuncAnimation(fig, animate, frames=len(train_losses), repeat=False, blit=True)
    plt.show()


# 测试模型
def model_eval(model, test_tensor):
    # 在测试集上进行评估
    test_tensor_stack = [torch.stack(tensors) for tensors in zip(*test_tensor)]
    test_dataset = TensorDataset(*test_tensor_stack)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, drop_last=True)
    # 将模型移动到GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # print("模型信息：", model)
    correct = 0
    total = 0
    with torch.no_grad():
        # 1
        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(test_loader):
            # 正向传播
            input_ids = input_ids.squeeze(1).to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = labels.squeeze().to(device)
            # 2
            # for i, data in enumerate(test_loader):
            #     # data = (input_ids, token_type_ids, attention_mask, labels)
            #     # 正向传播
            #     input_ids = data[0].squeeze(1).to(device)
            #     token_type_ids = data[1].squeeze(1).to(device)
            #     attention_mask = data[2].squeeze(1).to(device)
            #     labels = data[3].squeeze(1).to(device)
            # 3
            # for input_ids, token_type_ids, attention_mask, labels in test_loader:
            #     input_ids = input_ids.squeeze()  # 二维变成一维
            #     token_type_ids = token_type_ids.squeeze()
            #     attention_mask = attention_mask.squeeze()
            #     labels = labels.squeeze()

            labels = one_hot_encoding(labels, 2)
            labels = torch.argmax(labels, dim=1).cuda()  # 将one-hot标签转为整数标签

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


            # 这里需要深究一下
            # predicted = torch.argmax(outputs.logits, dim=1).item()
            # predicted = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1].item()
            # predictions = torch.argmax(outputs.logits)
            # predictions = torch.round(torch.sigmoid(outputs.logits))
            # predictions = torch.sigmoid(outputs.logits).round()

            # 这里不能加item()来取出预测值，因为验证的时候是用一个Batch_size大小的样本同时预测的，一个批次就是16个样本
            predictions = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1]  # 使用阈值0.5将概率转为二元标签

            print("输入预测标签为：", predictions)
            predictions_str = str(predictions.cpu().numpy().tolist())
            logging.info("输入预测标签为：" + predictions_str)
            labels_list = labels.tolist()
            print("输入实际标签为：", labels_list)
            logging.info("输入实际标签为：" + str(labels_list))
            # print("输入实际标签为：", labels.size(0))
            # print("输入实际标签2为：", labels.shape[0])
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print('Test accuracy:', accuracy)
    logging.info('Test accuracy: %f', accuracy)


# 定义预测函数
def predict(text, model, tokenizer):
    # 编码文本
    input_ids = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt').input_ids.cuda()
    model.cuda()
    # 预测类别
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # 不同于验证模型时不能用item()来取预测值，这里因为文本只有一条，所以预测值只有一个值，所以可以直接用item()来取出值
        # 但是注意：如果同时输入多条待预测文本，这里就不能用item()
        predicted = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1].item()
        # _, predicted = torch.max(outputs.logits, dim=1)
    return predicted


# 主函数入口
def main():
    # 获取当前时间
    local_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_filename = './运行日志/output_{}.log'.format(local_time)
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.StreamHandler(),  # 输出到终端
            logging.FileHandler(log_filename, mode='w', encoding='utf-8')  # 输出到文件
        ]
    )
    # 输出日志
    logging.info('——————程序运行开始——————')
    logging.info("——————————————————————")
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print('GPU is available')
        logging.info('GPU is available')
    else:
        print('GPU is not available')
        logging.info('GPU is not available')
    src_path = "./data/Rumor_Dataset.zip"
    target_path = "./data//Chinese_Rumor_Dataset-master"
    all_data_list = data_preprocessing(src_path, target_path)
    # 装载数据的根目录，负责存放数据txt文本
    data_list_path = "./data/"
    # 所有数据的txt文本路径
    all_data_path = data_list_path + "all_data.txt"
    # 数据字典
    dict_path = data_list_path + "dict.txt"
    # 创建字典
    create_dict(all_data_path, all_data_list, dict_path)
    # 划分训练集和测试集
    create_data_list(data_list_path)

    train_path = './data/train_list.txt'
    eval_path = './data/eval_list.txt'
    # tokenizer = BertTokenizer('gdrive/My Drive/Colab Notebooks/vocab.txt', do_lower_case=True)
    # tokenizer = BertTokenizer()
    bert = './models/chinese-bert-wwm-ext'
    # bert = './models/bert-base-chinese'
    # 配置文件
    # config = BertConfig.from_pretrained('bert-large-uncased', output_hidden_states=True, hidden_dropout_prob=0.2,
    #                                     attention_probs_dropout_prob=0.2)
    # 用BertTokenizer来作为模型加载
    # 加载模型和分词器
    # 加载预训练的BERT模型和中文分词器，并返回一个BertTokenizer对象，可以用于对中文文本进行分词和编码
    tokenizer = BertTokenizer.from_pretrained(bert, num_classes=2)
    model = BertForSequenceClassification.from_pretrained(bert)
    # 读取训练集和测试集
    # list: 2963, 里面是一个包含了4个元素的元组，四个元素中，有三个是张量，还有一个是整数
    splitting_dataset(all_data_path)
    train_data = './data/train_data.txt'
    test_data = './data/test_data.txt'
    train_tensor = read_data(train_data, tokenizer)
    test_tensor = read_data(test_data, tokenizer)
    # train_tensor = read_data(train_path, tokenizer)
    # test_tensor = read_data2(eval_path, tokenizer)

    trained_model = model_train(model, train_tensor)
    model_eval(trained_model, test_tensor)
    # 测试预测函数
    # text = '这个手机真的很好用'
    text = '有着800多年历史的克里姆林宫，再次迎来中国国家主席。当地时间3月21日下午，俄罗斯总统普京在大克里姆林宫二层乔治大厅为习近平主席举行隆重的欢迎仪式，随后两国元首举行会谈'
    predicted_label = predict(text, trained_model, tokenizer)
    labels = {0: "谣言", 1: "非谣言"}
    if predicted_label == 1:
        print(f'Predicted label: {predicted_label}')
        print("此微博博文为：", labels[predicted_label])
        # 输出日志
        logging.info(f'Predicted label: {predicted_label}')
        logging.info("此微博博文为：" + labels[predicted_label])
    logging.info("——————————————————————")
    logging.info('——————程序运行结束——————')


if __name__ == '__main__':
    # 设置训练超参数
    Batch_size = 16
    Epochs = 2  # 4
    Learning_rate = 2e-5
    main()

