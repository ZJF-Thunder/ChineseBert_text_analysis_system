from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch
# 准备数据
texts = ['某些食品添加剂含有致癌物质',
         '口罩会导致二氧化碳中毒',
         '喝酒后吃药会中毒',
         '狗能够感染新冠病毒']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
# padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
x = padded_sequences[:, :-1]
y = padded_sequences[:, 1:]

vocab_size = len(tokenizer.word_index) + 1

# 构建模型
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length - 1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 训练模型
model.fit(x, y, epochs=100)


# 生成新文本
def generate_text(model, tokenizer, max_length, seed_text, n_words):
    for _ in range(n_words):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length - 1, padding='post')

        prob = model.predict(sequence)[0]
        y_pred = np.random.choice(range(vocab_size), size=1, p=prob)[0]
        pred_word = tokenizer.index_word[y_pred]

        seed_text += ' ' + pred_word
    return seed_text


# 使用模型生成新的谣言文本
seed_text = '食品添加剂可以导致'
generated_text = generate_text(model, tokenizer, max_length, seed_text, 10)
print(generated_text)
