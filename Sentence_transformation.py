"""
功能：用chinesebert模型来进行句式转换，
通过掩码模型将句子中的词或字替换成其他词或字，以达到扩充数据集的目的
"""
import jieba
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random


def generate_sentences(text, label, num_sentences=3):
    # 将文本分词、编码为 BERT 所需的格式
    input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).cuda()
    # 在文本中选择一个随机词
    mask_token_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    mask_token = tokenizer.decode(input_ids[0, mask_token_index]).replace(' ', '')

    # 用 ChineseBERT 生成新的句子
    with torch.no_grad():
        output = model(input_ids)
        prediction_scores = output[0]
        masked_token_prediction = prediction_scores[0, mask_token_index]
        _, top_indices = torch.topk(masked_token_prediction, k=num_sentences)

    # 将生成的句子添加到列表中
    sentences = []
    for idx in top_indices:
        print("替换词的token id:", str(idx))
        print("替换词为：", tokenizer.decode([idx]))
        new_sentence = text.replace(mask_token, tokenizer.decode([idx]))
        print(new_sentence)
        print("——————————————————————————————————————————————————")
        sentences.append(label + "\t" + new_sentence + "\n")
    return sentences


# 将原始文本随机标记上"[MASK]"
def replace_with_mask(text):
    # 使用jieba进行高级别的精确分词
    words = jieba.lcut(text, cut_all=False)
    # 调用模型的分词工具
    # words = tokenizer.tokenize(text,
    #                            add_special_tokens=False,
    #                            do_basic_tokenize=False)

    # 随机定义需要掩码的字词索引
    replace_index = random.randint(0, len(words) - 2)
    # 将字词替换成"[MASK]"
    words[replace_index] = "[MASK]"
    masked_text = " ".join(words)
    # 先编码得到mask的特殊id，以供模型可以准确识别，并在句首和句尾添加特殊标志，然后解码恢复成带mask的文本,
    new_text = tokenizer.decode(tokenizer.encode(masked_text))
    # return new_text
    return masked_text


# 扩充数据集
def augmented_data():
    # 为每个文本生成新的句子
    augmented_dataset = []
    # 读取文本文件中的数据，然后扩充数据，得到augmented_dataset列表
    with open('./data/all_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            text = replace_with_mask(text)
            # 默认生成三个句子
            new_sentences = generate_sentences(text, label)
            augmented_dataset.extend(new_sentences)
    return augmented_dataset


# 合并数据集
def Merge_datasets(augmented_dataset):
    # 将扩充的文本写进文件
    add_sentence_txt = './data/add_sentence.txt'
    with open(add_sentence_txt, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(add_sentence_txt, 'a', encoding='utf-8') as f:
        for data in augmented_dataset:
            f.write(data)

    # 读取未扩充前的数据到data2列表中
    data_path = './data/all_data.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        data2 = f.readlines()
    print(data2)

    # 将原数据列表和扩充数据列表连接成一个新的列表
    all_data_list2 = augmented_dataset + data2
    # 将文本打乱顺序
    random.shuffle(all_data_list2)
    # 返回随机打乱的数据列表
    all_data_path2 = "./data/add_all_data2.txt"
    # 清空文本的数据，并往all_data_path的txt文本中写入所有文本数据
    with open(all_data_path2, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(all_data_path2, 'a', encoding='utf-8') as f:
        for data in all_data_list2:
            f.write(data)
    print("数据文本生成成功！")


if __name__ == '__main__':
    # bert = './models/bert-base-chinese'
    bert = './models/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(bert, num_labels=2)
    model = BertForMaskedLM.from_pretrained(bert)
    model.eval()
    model.cuda()
    data = augmented_data()
    Merge_datasets(data)
