"""
功能：对微调后并且保存的模型进行真实文本分类测试，非客户端版
"""
import torch
import jieba
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig


def predict(text, model, tokenizer):
    predicted_label, predicted_prob = predicted(text, model, tokenizer)
    # 另一个样本的标签及其概率
    other_prob = 1 - predicted_prob
    other_prob_label = 1
    # 定义标签
    labels = {0: "谣言", 1: "非谣言"}
    if predicted_label == 1:
        other_prob_label = 0
        print(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
              f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")
    else:
        print(f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
              f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")


# 预测模型
def predicted(text, model, tokenizer):
    """
    :param text: 待分类文本
    :param model: 训练完成并且保存的模型
    :param tokenizer: 文本分类器
    :return: 预测得到的标签及其概率
    """
    # 使用jieba进行高级别的精确分词
    words = jieba.lcut(text, cut_all=False)
    text = ' '.join(words)  # 将分词结果用空格拼接
    # 将文本转换成数字编码，这里是直接调用tokenizer类，
    # 其实内部处理分词的函数还是tokenizer.tokenize()
    # 并且内部encode的函数也是tokenizer.encode_plus()函数
    # input_ids = tokenizer(text,
    #                       padding=True,
    #                       truncation=True,
    #                       max_length=256,
    #                       return_tensors='pt',
    #                       do_basic_tokenize=False).input_ids.cuda()

    # # 这里可以直接调用tokenizer.encode_plus()函数来将token变成token id
    input_ids = tokenizer.encode_plus(text,
                                      padding=True,
                                      truncation=True,
                                      max_length=256,
                                      return_tensors='pt',
                                      is_split_into_words=True).input_ids.cuda()

    # 将模型移动到同一设备上
    model.cuda()
    # # 预测类别
    # model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # 使用softmax将输出转换成概率分布
        probs = torch.softmax(outputs.logits, dim=1)
        # 输出概率列表：probs.tolist()[0]
        # 获取最大概率对应的类别标签及其对应的概率
        pred_label = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_label].item()

    return pred_label, pred_prob


if __name__ == '__main__':
    bert = './models/chinese-bert-wwm-ext'
    # 加载自己保存后的config文件
    config = BertConfig.from_pretrained("my_chinesebert_config/config.json", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
    """加载自己保存的模型，多种模型对比"""
    # model_path = './models/chinese-bert-wwm-ext'  # 未经过微调的原始模型1
    # model_path = './models/bert-base-chinese'  # 未经过微调的原始模型2
    # model_path = './模型保存/chinesebert.pth'  # 微调的最早期的模型
    # model_path = './模型保存/ChineseBert_2023-03-29_16-27-07_0.949.pt'
    # model_path = './模型保存/ChineseBert_2023-03-25_17-10-39_0.95.pt'
    # model_path = './模型保存/ChineseBert_2023-04-05_16-30-38_0.9970.pt'
    model_path = './模型保存/ChineseBert_2023-04-07_20-12-29_0.999.pt'
    trained_model = BertForSequenceClassification.from_pretrained(model_path, config=config)

    # 假设你要对以下句子进行分类
    text1 = '有着800多年历史的克里姆林宫，再次迎来中国国家主席。当地时间3月21日下午，' \
            '俄罗斯总统普京在大克里姆林宫二层乔治大厅为习近平主席举行隆重的欢迎仪式，随后两国元首举行会谈'
    text2 = '3月24日，网传重庆永川区五米阳光小区一对夫妻发生矛盾，引发火灾。视频中，高层住宅楼上下两层房屋燃起大火，' \
            '浓烟蹿至楼顶，有东西掉落。'
    text3 = '二楼白布裹尸”“后山发现带血衣被”“后山发现实验室”“化粪池发现碎骨”“被光头老师杀害”“被化学老师用药水化掉”“' \
            '被人带进医院割去器官后抛尸河内”“SUV汽车运人”“录音笔已找到'
    predict(text3, trained_model, tokenizer)
