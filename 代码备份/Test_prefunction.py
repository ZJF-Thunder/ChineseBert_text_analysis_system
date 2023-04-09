# 对Test_model.py文件的函数进行验证和测试
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

bert = './models/chinese-bert-wwm-ext'
# 加载配置文件
config = BertConfig.from_pretrained(bert, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
trained_model = BertForSequenceClassification.from_pretrained('../模型保存/ChineseBert_2023-03-24_18-39-56.pt',
                                                              config=config)


# 定义一个预测函数
def predict(text, model, tokenizer):
    # 使用分词器将文本转换为token
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # 将token转换为pytorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # 使用模型进行预测
    outputs = model(input_ids)
    a = torch.sigmoid(outputs.logits)
    predicted = (torch.sigmoid(outputs.logits) >= 0.5).int()[:, 1].item()
    # 得到每个类别的概率分布
    logits = outputs[0]
    probs = torch.softmax(logits, dim=1)
    # 返回预测结果和相应的概率值
    pred_label = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0][pred_label].item()
    return pred_label, pred_prob


# 测试预测函数
# text = "这是一条测试文本"
text = '二楼白布裹尸”“后山发现带血衣被”“后山发现实验室”“化粪池发现碎骨”“被光头老师杀害”“被化学老师用药水化掉”“被人带进医院割去器官后抛尸河内”“SUV汽车运人”“录音笔已找到'
label, prob = predict(text, trained_model, tokenizer)
if label == 1:
    print(f"这条文本有{prob * 100:.2f}%的概率为非谣言")
else:
    print(f"这条文本有{prob * 100:.2f}%的概率为谣言")
