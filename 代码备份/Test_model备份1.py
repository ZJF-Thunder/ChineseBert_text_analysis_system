import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

bert = './models/chinese-bert-wwm-ext'
# 加载配置文件
config = BertConfig.from_pretrained(bert, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
trained_model = BertForSequenceClassification.from_pretrained('./模型保存/ChineseBert_2023-03-24_18-39-56.pt',
                                                              config=config)

# 假设你要对以下句子进行分类
sentence = "这是一条测试数据"
text = '有着800多年历史的克里姆林宫，再次迎来中国国家主席。当地时间3月21日下午，' \
       '俄罗斯总统普京在大克里姆林宫二层乔治大厅为习近平主席举行隆重的欢迎仪式，随后两国元首举行会谈'


def predict(text, model, tokenizer):
    # 编码文本
    input_ids = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt').input_ids.cuda()
    # 将模型移动到同一设备上
    model.cuda()
    # # 预测类别
    # model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # 不同于验证模型时不能用item()来取预测值，这里因为文本只有一条，所以预测值只有一个值，所以可以直接用item()来取出值
        # 但是注意：如果同时输入多条待预测文本，这里就不能用item()
        # 使用sigmoid激活函数
        # predicted = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1].item()

        # predicted = torch.argmax(outputs.logits, dim=1).item()
        # predicted = torch.max(outputs.logits, dim=1).item()

        # 使用bert模型自带的sotfmax激活函数
        # 得到每个类别的概率分布
        # 使用softmax将输出转换成概率分布
        probs = outputs.logits.softmax(dim=1)
        # 获取最大概率的类别标签
        # pred_label = probs.argmax(dim=1).item()
        pred_label = outputs.logits.argmax(dim=1).item()
        # 输出预测结果
        print(f"预测结果为: {pred_label}，概率分布为: {probs}")


        # 使用softmax将输出转换成概率分布
        probs = torch.softmax(outputs.logits, dim=1)
        # 输出概率
        print(f"Probabilities of non-rumor and rumor: {probs.tolist()[0]}")
        # 获取最大概率的类别标签及其对应的概率
        pred_label = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_label].item()
        # pred_prob = probs[0][pred_label].item()

        # 使用sigmoid没有softmax精确，使用softmax可以保证输出的概率分布总和为1，但sigmoid不能
        # probs = torch.sigmoid(outputs.logits)
        # print("概率大小：", probs)
        # pred_prob, pred_label = torch.max(probs, dim=1)

        # 输出预测结果和概率分布
        print(f"预测结果为: {pred_label}，概率分布为: {probs}")
    return pred_label, pred_prob
    # return predicted


predicted_label, predicted_prob = predict(text, trained_model, tokenizer)
other_prob = 1 - predicted_prob
other_prob_label = 1
labels = {0: "谣言", 1: "非谣言"}
if predicted_label == 1:
    other_prob_label = 0
    print(f"这条文本有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
          f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")
else:
    print(f"这条文本有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，"
          f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}")

