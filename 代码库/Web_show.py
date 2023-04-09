from flask import Flask, render_template, request

app = Flask(__name__)


# 首页路由
@app.route('/')
def index():
    return render_template('index.html')


# 分类和情感分析路由
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    predicted_label = predict(text, trained_model, tokenizer)
    return render_template('result.html', label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)













# 使用Flask框架来创建Web应用程序。下面是一个简单的示例，你可以根据自己的需求进行修改和扩展。
from flask import Flask, render_template, request
app = Flask(__name__)

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# 定义分类和情感分析函数
def predict(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    logits = model(input_ids)[0]
    preds = torch.argmax(logits, dim=1)
    return preds.item()

def analyze_sentiment(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    logits = model(input_ids)[0]
    probs = torch.softmax(logits, dim=1)
    prob_neg, prob_pos = probs[0]
    return prob_neg.item(), prob_pos.item()

# 定义路由
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    category = predict(text)
    if category == 0:
        category_name = '财经'
    elif category == 1:
        category_name = '教育'
    elif category == 2:
        category_name = '科技'
    else:
        category_name = '体育'
    prob_neg, prob_pos = analyze_sentiment(text)
    return render_template('result.html', category=category_name, prob_neg=prob_neg, prob_pos=prob_pos)

if __name__ == '__main__':
    app.run(debug=True)
