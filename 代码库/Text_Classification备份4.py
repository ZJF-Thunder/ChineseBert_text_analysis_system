"""
功能：加载微博谣言数据集，微调预训练模型并以此进行判断是否为谣言
"""
import json
import os
import random
import zipfile
import torch
import time
import datetime
import logging
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, BertConfig
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc
from torchmetrics.functional import hamming_distance


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
    all_data_path = "./data/all_data.txt"
    # 清空文本的数据，并往all_data_path的txt文本中写入所有文本数据
    with open(all_data_path, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(all_data_path, 'a', encoding='utf-8') as f:
        for data in all_data_list:
            f.write(data)
    print("数据文本生成成功！")
    logging.info("数据文本生成成功！")


# 获取谣言和非谣言的数据条数
def get_rumor_norumor_num(data_path):
    rumor_count = 0
    nonrumor_count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            label, text = line.strip().split('\t')
            if label == '0':
                rumor_count += 1
            else:
                nonrumor_count += 1
    print(f"数据集总数：{len(lines)} 条")
    print(f"谣言总数：{rumor_count} 条")
    print(f"非谣言总数：{nonrumor_count} 条")
    logging.info(f"数据集总数：{len(lines)} 条")
    logging.info(f"谣言总数：{rumor_count} 条")
    logging.info(f"非谣言总数：{nonrumor_count} 条")


# 按照9：1划分数据集
def splitting_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    train_data_path = './data/train_data.txt'
    test_data_path = './data/test_data.txt'
    with open(train_data_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(test_data_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    print("成功划分训练集和测试集！")
    logging.info("成功划分训练集和测试集！")
    print(f"训练集：{len(train_data)} 条")
    print(f"测试集：{len(test_data)} 条")
    logging.info(f"训练集：{len(train_data)} 条")
    logging.info(f"测试集：{len(test_data)} 条")

    return train_data_path, test_data_path


# WordPiece分词方法生成数据字典
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
def read_data(filename, tokenizer, max_seq_length=256):
    """
    :param max_seq_length:
    :param filename: 需要转换的数据的路径
    :return: 转换成模型所需的格式的数据
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        label, text = line.strip().split('\t')
        # 样例
        # encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, label=label)
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
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
    # 数据加载器，一个Batch(批次)等于样本总数/Batch_size
    # 在本样本中，训练数据一共3048个，batch_size为16，所以一共3048/16个batch，取整为190个batch
    # 所以train_loader的长度为190，它的作用是将3040个样本分割为190个batch，用于训练
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, drop_last=True)

    # 将数据加载到模型中进行微调
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # 定义交叉熵损失函数
    # criterion = torch.nn.BCEWithLogitsLoss()
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
    # 记录每一次迭代的损失
    train_losses = []
    print("————————————————模型训练开始————————————————")
    logging.info("————————————————模型训练开始————————————————")
    for epoch in range(Epochs):
        epoch_loss = 0  # 每一个Epoch的平均loss
        # 迭代一次就是迭代一个batch(批次)
        for batch, data in enumerate(train_loader):
            # data = (input_ids, token_type_ids, attention_mask, labels)
            # 正向传播
            input_ids = data[0].to(device).squeeze(1)
            token_type_ids = data[1].to(device).squeeze(1)
            attention_mask = data[2].to(device).squeeze(1)
            labels = data[3].to(device).squeeze()  # 形状为1维：（16，）
            # 将标签进行独热编码，变成长度为2的向量，和模型的输出层的长度一样，然后输入到模型中，并且调用二分类交叉损失函数
            labels = one_hot_encoding(labels, Num_labels)
            labels = labels.cuda()
            # 将模型的参数梯度清零，以免上一个epoch的梯度对当前epoch的训练产生影响
            optimizer.zero_grad()
            # 前向传播
            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            # 这是模型自带的损失函数,直接loss = outputs.loss调用即可
            loss = outputs.loss
            # 调用二元交叉熵损失函数计算损失,labels进行独热编码之后,本身就是浮点数,所以加不加float都一样
            # loss = criterion(outputs.logits, labels.float())
            # 调用交叉熵损失函数计算loss，这时候label不需要进行独热编码
            # loss = criterion(outputs.logits, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 调整学习率
            scheduler.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())

            if batch % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, loss.item()))
                logging.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, loss.item()))
        avg_epoch_loss = epoch_loss / len(train_loader)
        print("Epoch [{}/{}], Average Loss: {:.4f}".format(epoch + 1, Epochs, avg_epoch_loss))
        logging.info("Epoch [{}/{}], Average Loss: {:.4f}".format(epoch + 1, Epochs, avg_epoch_loss))

    end_time = time.time()
    total_time = end_time - start_time
    print("————————————————模型训练完成————————————————")
    print("训练开始时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print("训练结束时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print(f"模型训练总时间为: {total_time:.2f} 秒")
    logging.info("————————————————模型训练完成————————————————")
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

    # 绘制静态图
    # 清除控制台的输出train_losses
    # clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    # 散点图
    # plt.scatter(range(len(train_losses)), train_losses)
    # 折线图
    plt.clf()  # 首先清空当前图像
    plt.plot(range(len(train_losses)), train_losses, scalex=10)
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    img_save_dir = './images/training_loss'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    images = os.path.join(img_save_dir, "training_loss_{}.png".format(local_time))
    plt.savefig(images)
    # plt.show()

    # 绘制动态图
    fig, ax = plt.subplots()
    ax.set(xlim=(0, len(train_losses)), ylim=(0, 1))

    def animate(i):
        ax.clear()
        ax.plot(range(i + 1), train_losses[:i + 1], label='Training Loss')
        ax.legend(loc='upper right')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
        plt.pause(0.1)
        # plt.savefig('figure.png')

    ani = FuncAnimation(fig, animate, frames=len(train_losses), repeat=True)
    # ani.save('animation2.gif', writer='pillow')
    # plt.show()  # 把注释去掉可以查看动态图
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


# 定义计算指标函数
def compute_metrics(y_true, y_pred):
    # 准确率、准确度
    accuracy = accuracy_score(y_true, y_pred)
    # 精确度、精度：所有被分类器正确分类的正样本占所有被分类器分类为正样本的样本数的比例
    precision = precision_score(y_true, y_pred, average='macro')
    # 召回率：被正确分类的正样本占所有实际正样本的比例
    recall = recall_score(y_true, y_pred, average='macro')
    # F1：精确度和召回率的调和平均值
    f1 = f1_score(y_true, y_pred, average='macro')
    # AUC值：ROC曲线下的面积,取值范围在0到1之间，越接近1代表模型的性能越好
    auc = roc_auc_score(y_true, y_pred)
    result = {'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'f1_score': f1,
              'AUC': auc}
    return result


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
    # 存放真实标签和预测标签
    label_true = []
    label_pred = []
    print("————————————————模型测试开始————————————————")
    logging.info("————————————————模型测试开始————————————————")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # data = (input_ids, token_type_ids, attention_mask, labels)
            # 正向传播
            input_ids = data[0].to(device).squeeze(1)  # 降维度
            token_type_ids = data[1].to(device).squeeze(1)
            attention_mask = data[2].to(device).squeeze(1)
            labels = data[3].to(device).squeeze()
            print("输入实际标签为：", labels)
            logging.info("输入实际标签为：" + str(labels.tolist()))
            # labels = one_hot_encoding(labels, 2)
            # labels = torch.argmax(labels, dim=1).cuda()  # 将one-hot标签转为整数标签

            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            # 这里不能加item()来取出预测值，因为验证的时候是用一个Batch_size大小的样本同时预测的，一个批次就是16个样本
            # 这里predictiongs不是一个样本，是一个有着16个样本预测标签的类似列表或者张量
            predictions = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1]  # 使用阈值0.5将概率转为二元标签
            predictions = torch.argmax(torch.sigmoid(outputs.logits), dim=1).cuda()
            predictions = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1).cuda()
            print("输入预测标签为：", predictions)
            logging.info("输入预测标签为：" + str(predictions.cuda().tolist()))

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # sklearn的指标  全部tensor转换成list
            """extend和append的区别：
            extend是直接在后面追加所有元素，如果追加的是一个列表，则是直接追加这个列表的所有值
            append是在后面追加一个单独的元素，这个元素可以是一个列表或者一个值"""
            label_true.extend(labels.cuda().tolist())
            label_pred.extend(predictions.cuda().tolist())

    print("————————————————模型测试结束————————————————")
    logging.info("————————————————模型测试结束————————————————")
    accuracy = correct / total
    print("总测试样本：{}个".format(total))
    print("预测正确样本:{}个".format(correct))
    print('测试准确度:', accuracy)
    logging.info('总测试样本: %d个', total)
    logging.info('预测正确样本: %d个', correct)
    logging.info('测试准确度: %f', accuracy)

    # sklearn的指标
    metrics = compute_metrics(label_true, label_pred)
    print("-----------------------------")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"AUC Score: {metrics['AUC']}")
    print("-----------------------------")
    print(classification_report(label_true, label_pred))
    print("-----------------------------")
    print("-----------混淆矩阵------------")
    print(confusion_matrix(label_true, label_pred))
    print("-----------------------------")

    logging.info("-----------------------------")
    logging.info(f"Accuracy: {metrics['accuracy']}")
    logging.info(f"Precision: {metrics['precision']}")
    logging.info(f"Recall: {metrics['recall']}")
    logging.info(f"F1 Score: {metrics['f1_score']}")
    logging.info(f"AUC Score: {metrics['AUC']}")
    logging.info("-----------------------------")
    logging.info("\n" + classification_report(label_true, label_pred))
    logging.info("-----------------------------")
    logging.info("-----------混淆矩阵------------")
    logging.info("\n" + str(confusion_matrix(label_true, label_pred)))
    logging.info("-----------------------------")

    # 计算average_precision_score的值
    average_precision = average_precision_score(label_true, label_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    # 计算汉明损失
    hamming_loss = hamming_distance(torch.tensor(label_pred), torch.tensor(label_true))
    print(f"Hamming Loss: {hamming_loss}")

    # 计算不同概率阈值的精度-召回率对
    precision, recall, thresholds = precision_recall_curve(label_true, label_pred)
    # 计算AUC-PR
    auc_PR = auc(recall, precision)
    # 绘制PR曲线::纵轴是精度（precision），横轴是召回率（recall）
    plt.clf()  # 首先清空当前图像
    plt.plot(recall, precision, lw=2, color='blue', label='AUC-PR = %0.4f' % auc_PR)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="lower right")
    img_save_dir = './images/PR曲线'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    # 获取当前时间
    local_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    images = os.path.join(img_save_dir, "PR_{}.png".format(local_time))
    plt.savefig(images)
    # plt.show()

    # 计算假阳性率（FPR）和真阳性率（TPR）和阈值（thresholds）
    FPR, TPR, thresholds = roc_curve(label_true, label_pred)
    # 计算AUC-ROC
    auc_ROC = auc(FPR, TPR)
    # 绘制ROC曲线:纵轴是真阳率（true positive rate），横轴是假阳率（false positive rate）
    plt.clf()  # 首先清空当前图像
    plt.plot(FPR, TPR, lw=2, color='red', label='AUC-ROC = %0.4f' % auc_ROC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    img_save_dir = './images/ROC曲线'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    # 获取当前时间
    local_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    images = os.path.join(img_save_dir, "ROC_{}.jpg".format(local_time))
    plt.savefig(images)
    # plt.show()


# 定义预测函数
def predicted(text, model, tokenizer):
    # 将文本转换成数字编码
    input_ids = tokenizer(text,
                          padding=True,
                          truncation=True,
                          max_length=Max_seq_length,
                          return_tensors='pt').input_ids.cuda()
    model.cuda()
    # 预测类别
    with torch.no_grad():
        outputs = model(input_ids)
        # 不同于验证模型时不能用item()来取预测值，这里因为文本只有一条，所以预测值只有一个值，所以可以直接用item()来取出值
        # 但是注意：如果同时输入多条待预测文本，这里就不能用item()
        # predicted = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1].item()
        # 使用softmax将输出转换成概率分布
        probs = torch.softmax(outputs.logits, dim=1)
        # 输出概率列表：probs.tolist()[0]
        # 获取最大概率对应的类别标签及其对应的概率
        pred_label = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_label].item()

    return pred_label, pred_prob


# 调用预测模型来预测文本类别
def Text_predict(trained_model, tokenizer):
    print("————————————————预测文本类别————————————————")
    logging.info("————————————————预测文本类别————————————————")
    text = '有着800多年历史的克里姆林宫，再次迎来中国国家主席。当地时间3月21日下午，俄罗斯总统普京在大克里姆林宫二层乔治大厅为习近平主席举行隆重的欢迎仪式，随后两国元首举行会谈'
    predicted_label, predicted_prob = predicted(text, trained_model, tokenizer)
    # 另一个样本的标签及其概率
    other_prob = 1 - predicted_prob
    other_prob_label = 1
    # 定义标签
    labels = {0: "谣言", 1: "非谣言"}
    if predicted_label == 1:
        other_prob_label = 0
        print(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
              f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")
        logging.info(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
                     f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")
    else:
        print(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
              f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")
        logging.info(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
                     f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")


# 检查GPU是否可用，不可用则退出程序
def cuda_is_available():
    if torch.cuda.is_available():
        print('GPU is available')
        logging.info('GPU is available')
    else:
        print('GPU is not available')
        logging.info('GPU is not available')
        print("————————正在退出程序——————————")
        logging.info("————————正在退出程序——————————")
        exit(1)


# 日志配置
def log_config():
    # 获取当前日期
    today = datetime.date.today()
    date_str = today.strftime('%Y-%m-%d')
    # 获取当前具体时间
    local_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    # 创建日志文件路径
    log_file_path = os.path.join('./运行日志/', f"{date_str}日志")
    # 判断文件夹是否存在，如果不存在则创建
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
        print(f"文件夹 {log_file_path} 创建成功！")
    else:
        print(f"文件夹 {log_file_path} 已存在！")
    log_filename = '{}/output_{}.log'.format(log_file_path, local_time)
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.StreamHandler(),  # 输出到终端
            logging.FileHandler(log_filename, mode='w', encoding='utf-8')  # 输出到文件
        ]
    )


# 主函数入口
def main():
    log_config()
    # 输出日志
    logging.info('——————程序运行开始——————')
    logging.info("——————————————————————")
    # 输出训练超参数
    logging.info("——————输出训练参数——————")
    logging.info("Batch_size：%s", Batch_size)
    logging.info("Epochs：%s", Epochs)
    logging.info("Learning_rate：%s", Learning_rate)
    logging.info("max_seq_length：%s", Max_seq_length)
    logging.info("num_labels：%s", Num_labels)
    logging.info("——————输出数据信息——————")
    # 检查GPU是否可用
    cuda_is_available()
    # 待解压文件路径
    src_path = "./data/Rumor_Dataset.zip"
    # 解压之后文件路径
    target_path = "./data//Chinese_Rumor_Dataset-master"
    # 所有数据的txt文本路径
    all_data_path = "./data/all_data.txt"
    # 将数据集解压，并将json文件数据处理保存成txt文本  没有扩充之前的数据集
    # data_preprocessing(src_path, target_path)

    # 划分训练集和测试集
    # train_data, test_data = splitting_dataset(all_data_path)
    # 获取谣言和非谣言的数据条数
    get_rumor_norumor_num("./data/all_data2.txt")
    # 用扩充之后的数据集划分训练集和测试集，all_data2.txt为扩充之后的数据集
    train_data, test_data = splitting_dataset("./data/all_data2.txt")

    # # 可以舍弃
    # # 装载数据的根目录，负责存放数据txt文本
    # data_list_path = "./data/"
    # # 创建数据字典
    # dict_path = "./data/dict.txt"
    # # create_dict(all_data_path, dict_path)
    # # 划分训练集和测试集 抛弃
    # # create_data_list(data_list_path)
    # train_path = './data/train_list.txt'
    # eval_path = './data/eval_list.txt'

    # 可以在这里指定字典的路径
    # tokenizer = BertTokenizer('gdrive/My Drive/Colab Notebooks/vocab.txt')

    # 定义预训练模型
    bert = './models/chinese-bert-wwm-ext'
    # bert = './models/bert-base-chinese'
    # 加载原始config文件
    config = BertConfig.from_pretrained(bert, num_labels=Num_labels)
    """# 加载配置文件，可以在参数中修改配置文件
    # config = BertConfig.from_pretrained(bert, output_hidden_states=True, hidden_dropout_prob=0.2,
    # 也可以单独定义配置文件中的参数，以此来修改配置文件，然后保存
    # attention_probs_dropout_prob=0.2)
    # 如果修改了，就保存修改后的配置文件，如果没有修改参数则和原始参数相同"""
    config.save_pretrained('my_chinesebert_config')
    # 用BertTokenizer来作为模型加载
    # 加载预训练的BERT模型和中文分词器，并返回一个BertTokenizer对象，可以用于对中文文本进行分词和编码
    tokenizer = BertTokenizer.from_pretrained(bert, num_classes=Num_labels)
    model = BertForSequenceClassification.from_pretrained(bert, config=config)

    # 读取训练集和测试集
    train_tensor = read_data(train_data, tokenizer, max_seq_length=Max_seq_length)
    test_tensor = read_data(test_data, tokenizer, max_seq_length=Max_seq_length)

    # # 训练模型
    # trained_model = model_train(model, train_tensor)
    # # 测试模型
    # model_eval(trained_model, test_tensor)
    # # 预测函数
    # Text_predict(trained_model, tokenizer)

    """用于测试已有模型"""
    trained_model_path = './模型保存/ChineseBert_2023-04-05_16-30-38_0.997.pt'
    model = BertForSequenceClassification.from_pretrained(trained_model_path, config=config)
    model_eval(model, test_tensor)
    Text_predict(model, tokenizer)

    logging.info("——————————————————————")
    logging.info('——————程序运行结束——————')


if __name__ == '__main__':
    # 设置训练超参数
    Batch_size = 12
    Epochs = 1  # 4
    Learning_rate = 2e-5
    Num_labels = 2
    Max_seq_length = 256
    main()
