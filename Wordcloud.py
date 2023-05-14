# 导入需要的库
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba.analyse
import codecs


# 生成词云图
def get_wordcloud(filepath):
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 读入停用词表
    stop_words = set()
    data = []
    # 定义标签名，以此获取生成词云图名称
    label_name = 0
    for line in lines:
        label, text = line.strip().split('\t')
        if label == '0':
            label_name = 'rumor_wordcloud'
        elif label == '1':
            label_name = 'non_rumor_wordcloud'
        # 使用jieba进行高级别的精确分词，并去除停用词
        words = jieba.lcut(text, cut_all=False)
        with codecs.open('stop_words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
        tokens = []
        for word in words:
            # 去除停用词和空格
            if word not in stop_words and word.strip():
                tokens.append(word)
        text = ' '.join(tokens)  # 将分词结果用空格拼接
        print(text)
        data.append(text)

    # 加载背景图片
    background_image = np.array(Image.open('./my_js/bg3.jpg'))

    # 生成词云图
    wordcloud = WordCloud(
        background_color=None,
        max_words=200, width=800, height=800,
        mask=background_image,
        stopwords=stop_words,
        font_path='font.ttf',  # 用于显示中文的字体文件路径
        # max_font_size=50, # 设置字体大小
        random_state=42
    ).generate(str(data))

    # 显示词云图
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # plt.show()

    # 保存词云图
    wordcloud.to_file(f'./images/WordCloud/{label_name}.png')


if __name__ == '__main__':
    get_wordcloud('./data/non_rumor_data.txt')
    get_wordcloud('./data/rumor_data.txt')
    print('done')
