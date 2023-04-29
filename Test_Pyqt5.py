import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt5.QtGui import QBrush, QPixmap, QFont, QPalette, QColor
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from Test_model import predicted


class TextAnalyzer(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('文本分析系统')
        self.setGeometry(300, 300, 500, 500)
        self.resize(1000, 700)
        # 创建文本输入框、分析按钮、结果显示框和标签
        self.textEdit = QTextEdit()
        self.resultLabel = QLabel("等待分析")
        self.analyzeBtn = QPushButton('分析')
        self.resultTextEdit = QTextEdit()
        # self.setStyleSheet(
        #     "background-image: url(./my_js/bg.jpg); background-position: center center; background-size: cover;")

        # 创建一个QPixmap对象，并设置为背景图片  设置整个窗口的背景图片
        pixmap = QPixmap('./my_js/win11.jpg')
        pixmap = pixmap.scaled(self.size())  # 将背景图片大小设置为和显示框一样大
        # 创建一个QPalette对象，并将背景图片应用到QPalette.Background属性中
        palette = self.palette()
        palette.setBrush(self.backgroundRole(), QBrush(pixmap))
        self.setPalette(palette)

        # 将所有控件的背景设置为透明
        self.textEdit.setAutoFillBackground(False)
        self.analyzeBtn.setAutoFillBackground(False)
        self.resultLabel.setAutoFillBackground(False)
        self.resultTextEdit.setAutoFillBackground(False)

        # 将所有控件的样式设置为透明
        self.textEdit.setStyleSheet("background-color: rgba(0,0,0,0);")
        # self.analyzeBtn.setStyleSheet("background-color: rgba(0,0,0,0);")
        self.resultLabel.setStyleSheet("background-color: rgba(0,0,0,0);")
        self.resultTextEdit.setStyleSheet("background-color: rgba(0,0,0,0);")
        # 为分析按钮单独设置样式
        # self.analyzeBtn.setStyleSheet("QPushButton { background-color: red; color: white; }")
        # self.analyzeBtn.setStyleSheet("background-color: red; color: white")
        self.analyzeBtn.setStyleSheet("QPushButton {background-color: #4CAF50; "
                                      "border: none;"
                                      "color: white; "
                                      "padding: 10px 20px; "
                                      "text-align: center; "
                                      "text-decoration: none;"
                                      "display: inline-block;"
                                      "font-size: 16px;"
                                      "margin: 4px 2px;"
                                      "cursor: pointer;"
                                      "border-radius: 8px;"
                                      "box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4);}"
                                      "QPushButton:hover "
                                      "{background-color: #3e8e41;}"
                                      "QPushButton:pressed "
                                      "{background-color: #145826;box-shadow: none;}")
        '''background - color：设置按钮的背景颜色为  # 4CAF50（一种绿色）。
        border：设置按钮的边框为空。
        color：设置按钮上的文本颜色为白色。
        padding：设置按钮内部的上下左右间距分别为10px和20px。
        text - align：设置按钮内部文本的对齐方式为居中。
        text - decoration：设置按钮内部文本的装饰方式为空（不带下划线）。
        display：设置按钮的显示方式为内联块级元素。
        font - size：设置按钮内部文本的字体大小为16px。
        margin：设置按钮外边距的上下左右间距分别为4px和2px。
        cursor：设置鼠标指针在按钮上的样式为手型。
        border - radius：设置按钮的圆角半径为8px。
        box - shadow：设置按钮的阴影效果，包括阴影的位置、大小、颜色和透明度。
        css
        Copy
        code
        QPushButton: hover
        {
            background - color:  # 3e8e41;
        }
        :hover：设置按钮在鼠标悬停时的样式。
        background - color：设置按钮背景颜色为  # 3e8e41（一种深绿色）。
        css
        Copy
        code
        QPushButton: pressed
        {
            background - color:  # 145826;
                box - shadow: none;
        }
        :pressed：设置按钮在被点击时的样式。
        background - color：设置按钮背景颜色为  # 145826（一种更深的绿色）。
        box - shadow：取消按钮的阴影效果，使其看起来像是被按下去的效果。
        
        :hover：设置按钮在鼠标悬停时的样式。
        background-color：设置按钮背景颜色为 #3e8e41（一种深绿色）。
        :pressed：设置按钮在被点击时的样式。
        background-color：设置按钮背景颜色为 #145826（一种更深的绿色）。
        box-shadow：取消按钮的阴影效果，使其看起来像是被按下去的效果。
        '''

        # 创建一个QFont对象，设置字体大小为15
        font = QFont()
        font.setPointSize(15)
        # 将QFont对象应用到QTextEdit控件中
        self.textEdit.setFont(font)
        self.resultLabel.setFont(font)
        self.resultTextEdit.setFont(font)

        # 给结果标签创建一个QPalette对象，并设置其颜色为红色
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor("red"))
        self.resultLabel.setPalette(palette)

        # 给结果显示框创建一个QPalette对象，并设置其颜色为红色
        palette2 = QPalette()
        palette2.setColor(QPalette.Text, QColor("red"))
        self.resultTextEdit.setPalette(palette2)

        # 给文本输入框创建一个QPalette对象，并设置其颜色为红色
        palette3 = QPalette()
        palette3.setColor(QPalette.Text, QColor("black"))
        # self.textEdit.setPalette(palette3)

        # 直接设置颜色，不需要用到QPalette控件 以下两种均可
        self.textEdit.setTextColor(QColor('black'))
        # 这个是设置输入框的颜色，不是输入文字的颜色
        # self.textEdit.setStyleSheet('color: white')

        # # 也可以使用QPalette控件实现更精细化的样式
        # # 创建QPalette对象
        # palette4 = QPalette()
        # # 设置颜色
        # palette4.setColor(QPalette.Base, QColor('white'))  # 设置文本框的背景颜色
        # palette4.setColor(QPalette.Text, QColor('blue'))  # 设置文本框的前景颜色
        # # 将QPalette对象应用到QTextEdit控件中
        # self.textEdit.setPalette(palette4)

        # 为分析按钮添加点击事件处理程序
        self.analyzeBtn.clicked.connect(self.analyzeText)

        # 创建垂直和水平布局，并将控件添加到布局中
        vBox = QVBoxLayout()
        hBox = QHBoxLayout()
        hBox.addWidget(self.textEdit)
        hBox.addWidget(self.analyzeBtn)
        vBox.addLayout(hBox)
        vBox.addWidget(self.resultLabel)
        vBox.addWidget(self.resultTextEdit)

        # 设置主布局
        self.setLayout(vBox)

        # 显示窗口
        self.show()

    def analyzeText(self):
        # 获取文本输入框的内容
        text = self.textEdit.toPlainText()
        self.resultLabel.setText('正在分析，请稍候...')
        QApplication.processEvents()  # 刷新界面，使QLabel文本立即更新

        predicted_label, predicted_prob = predicted(text, model, tokenizer)
        # 另一个样本的标签及其概率
        other_prob = 1 - predicted_prob
        other_prob_label = 1
        # 定义标签
        labels = {0: "谣言", 1: "非谣言"}
        if predicted_label == 1:
            other_prob_label = 0
            result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，" \
                     f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"
        else:
            result = f"这条微博有{predicted_prob * 100:.2f}%的概率为{labels[predicted_label]}，" \
                     f"有{other_prob * 100:.2f}%的概率为{labels[other_prob_label]}"

        # 显示结果
        # 更新QLabel文本
        self.resultLabel.setText("分析已完成")
        self.resultTextEdit.setText(result)


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

    app = QApplication(sys.argv)
    ex = TextAnalyzer()
    sys.exit(app.exec_())
