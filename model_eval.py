"""
功能：评估训练好的模型的性能指标
"""
import os
import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchmetrics.functional import hamming_distance
from Text_Classification import read_data


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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, drop_last=True)
    # 将模型移动到GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 预测正确的样本个数
    correct = 0
    # 总样本个数
    total = 0

    # 存放真实标签和预测标签
    label_true = []
    label_pred = []

    print("————————————————模型测试开始————————————————")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # 正向传播
            input_ids = data[0].to(device).squeeze(1)  # 降维度
            token_type_ids = data[1].to(device).squeeze(1)
            attention_mask = data[2].to(device).squeeze(1)
            labels = data[3].to(device).squeeze()
            print("输入实际标签为：", labels)

            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            predictions = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1).cuda()
            print("输入预测标签为：", predictions)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # sklearn的指标  全部tensor转换成list
            """extend和append的区别：
            extend是直接在后面追加所有元素，如果追加的是一个列表，则是直接追加这个列表的所有值
            append是在后面追加一个单独的元素，这个元素可以是一个列表或者一个值"""
            label_true.extend(labels.cuda().tolist())
            label_pred.extend(predictions.cuda().tolist())

    print("————————————————模型测试结束————————————————")

    accuracy = correct / total
    print("总测试样本：{}个".format(total))
    print("预测正确样本:{}个".format(correct))
    print('测试准确度:', accuracy)

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
    fpr, tpr, thresholds = roc_curve(label_true, label_pred)
    # 计算AUC-ROC
    auc_ROC = auc(fpr, tpr)
    # 绘制ROC曲线:纵轴是真阳率（true positive rate），横轴是假阳率（false positive rate）
    plt.clf()  # 首先清空当前图像
    plt.plot(fpr, tpr, lw=2, color='red', label='AUC-ROC = %0.4f' % auc_ROC)
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


if __name__ == '__main__':
    # 定义预训练模型
    bert = './models/chinese-bert-wwm-ext'
    # 加载自己保存的config文件
    config = BertConfig.from_pretrained("my_chinesebert_config/config.json", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
    test_tensor = read_data('./data/test_data.txt', tokenizer, max_seq_length=256)
    """加载自己保存的模型，多种模型对比"""
    # model_path = './models/chinese-bert-wwm-ext'  # 未经过微调的原始模型1
    # model_path = './models/bert-base-chinese'  # 未经过微调的原始模型2
    # model_path = './模型保存/chinesebert.pth'  # 微调的最早期的模型
    # model_path = './模型保存/ChineseBert_2023-03-29_16-27-07_0.949.pt'
    # model_path = './模型保存/ChineseBert_2023-03-25_17-10-39_0.95.pt'
    # model_path = './模型保存/ChineseBert_2023-04-05_16-30-38_0.9970.pt'
    model_path = './模型保存/ChineseBert_2023-04-07_20-12-29_0.999.pt'
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)

    # 测试模型
    model_eval(model, test_tensor)
