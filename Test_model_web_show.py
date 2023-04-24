"""
功能：网页中输入待测文本，用保存后的模型进行文本分类测试，客户端版
"""
import jieba
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
    # text = request.form['text']
    text = request.form.get('text')
    if text is None or text.strip() == '':
        massage = '请输入文本'
        return jsonify({'result': massage})
    # 调用模型，得到分类结果
    predicted_label, predicted_prob = predicted(text, model, tokenizer)
    other_prob = 1 - predicted_prob
    other_prob_label = 1
    labels = {0: "谣言", 1: "非谣言"}
    if predicted_label == 1:
        other_prob_label = 0
        result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，" \
                 f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"
    else:
        result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，" \
                 f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"
    return jsonify({'result': result})


# 模型预测函数
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

    # 这里可以直接调用tokenizer.encode_plus()函数来将token变成token id
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
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    # app.run(debug=True)
    app.run(debug=False)
