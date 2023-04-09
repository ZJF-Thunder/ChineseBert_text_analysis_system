from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# 初始化应用程序
app = Flask(__name__, template_folder="my_html", static_folder="my_js")


# 定义路由
@app.route('/')
def index():
    # return app.send_static_file('index5.html')
    return render_template("index5.html")


# 定义路由
@app.route('/predict', methods=["GET", "POST"])
def predict():
    # text = request.json['text']
    text = request.form['text']
    predicted_label, predicted_prob = predicted(text, model, tokenizer)
    other_prob = 1 - predicted_prob
    other_prob_label = 1
    labels = {0: "谣言", 1: "非谣言"}
    if predicted_label == 1:
        other_prob_label = 0
        result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"
    else:
        result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"
    return jsonify({'result': result})
    # return jsonify({'label': '谣言' if predicted_label == 0 else '非谣言'})


# 模型预测函数
def predicted(text, model, tokenizer):
    """
    :param text: 待分类文本
    :param model: 训练完成并且保存的模型
    :param tokenizer: 文本分类器
    :return: 预测得到的标签及其概率
    """
    # 编码文本
    input_ids = tokenizer(text,
                          padding=True,
                          truncation=True,
                          max_length=256,
                          return_tensors='pt').input_ids.cuda()
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


# @app.route('/predict', methods=["GET", "POST"])
# def predict():
#     text = request.form['text']
#     input_ids = tokenizer(text,
#                           padding=True,
#                           truncation=True,
#                           max_length=256,
#                           return_tensors='pt').input_ids.cuda()
#     with torch.no_grad():
#         outputs = model(input_ids)
#         probs = torch.softmax(outputs.logits, dim=1)
#         predicted_label = torch.argmax(probs, dim=1).item()
#
#     return jsonify({'label': '谣言' if predicted_label == 0 else '非谣言'})


if __name__ == '__main__':
    bert = './models/chinese-bert-wwm-ext'
    config = BertConfig.from_pretrained(bert, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
    model_path = './模型保存/ChineseBert_2023-03-25_17-10-39_0.95.pt'
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    app.run(debug=True)
