import json

import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig, BertForSequenceClassification
# 对应index2
app = Flask(__name__)

# 加载预训练好的中文Bert模型和tokenizer
# model = TFBertForSequenceClassification.from_pretrained("bert-base-chinese")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

bert = './models/chinese-bert-wwm-ext'
# 加载配置文件
config = BertConfig.from_pretrained(bert, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
model = BertForSequenceClassification.from_pretrained('./模型保存/ChineseBert_2023-03-25_17-10-39_0.95.pt',
                                                      config=config)


# 定义分类和情感分析函数
def classify_text(text):
    # 使用tokenizer对文本进行编码
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    # 使用模型进行分类和情感分析
    output = model(encoded_text["input_ids"], encoded_text["attention_mask"])
    logits = output.logits.numpy()[0]
    probs = [float(f"{i:.3f}") for i in torch.softmax(logits)]
    category = "正面" if logits[1] > logits[0] else "负面"
    return category, probs[0], probs[1]


# 定义路由函数来处理POST请求
@app.route("/classify", methods=["POST"])
def classify():
    # 获取POST请求中的文本数据
    text = request.form["text"]

    # 对文本进行分类和情感分析
    category, prob_neg, prob_pos = classify_text(text)

    # 将分类结果和情感概率作为JSON格式的响应返回给前端页面
    response = {
        "category": category,
        "prob_neg": prob_neg,
        "prob_pos": prob_pos,
    }
    return json.dumps(response)


# 定义路由函数来渲染前端页面
@app.route("/")
def index():
    return render_template("index.html")
