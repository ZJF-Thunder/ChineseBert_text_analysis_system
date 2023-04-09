import jieba
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random

tokenizer = BertTokenizer.from_pretrained('../models/bert-base-chinese')
model = BertForMaskedLM.from_pretrained('../models/bert-base-chinese')
model.eval()
model.cuda()


def generate_sentences(text, label, num_sentences=3):
    # 将文本分词、编码为 BERT 所需的格式
    input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False, do_basic_tokenize=True).cuda()
    # 在文本中选择一个随机词
    mask_token_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    mask_token = tokenizer.decode(input_ids[0, mask_token_index]).replace(' ', '')

    # mask_token = tokenizer.decode(input_ids[0, mask_token_index])
    # tokens = tokenizer.convert_ids_to_tokens([idx])
    # new_sentence = text.replace(mask_token, ''.join(tokens))

    # 用 BERT 生成新的句子
    with torch.no_grad():
        output = model(input_ids)
        # output.logits
        prediction_scores = output[0]
        masked_token_prediction = prediction_scores[0, mask_token_index]
        _, top_indices = torch.topk(masked_token_prediction, k=num_sentences)

    # 将生成的句子添加到列表中
    sentences = []
    for idx in top_indices:
        print("idx:", idx)
        # 用选择的单词替换掉原始文本中的 [MASK] 标记
        # tokens = str(tokenizer.convert_ids_to_tokens([idx]))
        # tokenizer.decode([idx])
        print(tokenizer.decode([idx]))
        print(mask_token)
        new_sentence = text.replace(mask_token, tokenizer.decode([idx]))
        print(new_sentence)
        sentences.append(label + "\t" + new_sentence + "\n")
        # sentences.append(new_sentence + '\n')
        print(sentences)
    return sentences


def generate_sentences2(text, label, num_sentences=3):
    # 将文本切分成单个的汉字
    # 这个就是encode_plus函数里面的分词操作
    tokens = tokenizer.tokenize(text, add_special_tokens=False, do_basic_tokenize=False)
    # 在文本中选择一个随机字
    tokens = list(text)
    mask_token_index = torch.randint(0, len(tokens), (1,)).item()
    print(mask_token_index)
    # mask_token = tokens[mask_token_index]
    # print(mask_token)

    # 将分词之后的汉字转换成对应的 ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids[mask_token_index] = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # input_ids[mask_token_index] = tokenizer.mask_token_id
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

    # 用 BERT 生成新的句子
    with torch.no_grad():
        output = model(input_ids)
        prediction_scores = output[0]
        masked_token_prediction = prediction_scores[0, mask_token_index]
        _, top_indices = torch.topk(masked_token_prediction, k=num_sentences)

    # 将生成的句子添加到列表中
    sentences = []
    for idx in top_indices:
        print("idx:", idx)
        # 用选择的汉字替换掉原始文本中的 mask_token
        new_tokens = tokens.copy()
        new_tokens[mask_token_index] = tokenizer.convert_ids_to_tokens([idx])[0]
        print(tokenizer.decode([idx]))
        # print(mask_token)
        print(new_tokens[mask_token_index])
        new_sentence = ''.join(new_tokens)
        print(new_sentence)
        sentences.append(label + "\t" + new_sentence + "\n")
    return sentences


# 将原始文本随机标记上"[MASK]"
def replace_with_mask(text):
    # 分词
    # 试一下jieba的分词
    # 使用jieba进行高级别的精确分词
    words = jieba.lcut(text, cut_all=False)
    # text = ' '.join(words)  # 将分词结果用空格拼接,变成有空格的str
    # print(words)
    # print(text)
    # 调用模型的分词工具
    # words = tokenizer.tokenize(text,
    #                            add_special_tokens=False,
    #                            do_basic_tokenize=False)
    # 随机定义需要掩码的字词索引
    replace_index = random.randint(0, len(words) - 2)
    # replace_index = random.randint(0, len(tokenized_text) - 2)
    print(replace_index)
    # 将字词替换成"[MASK]"
    # tokenized_text[replace_index] = "[MASK]"
    words[replace_index] = "[MASK]"
    # masked_text = " ".join(tokenized_text)
    masked_text = " ".join(words)
    # 先编码得到mask的特殊id，以供模型可以准确识别，并在句首和句尾添加特殊标志，然后解码恢复成带mask的文本,
    new_text = tokenizer.decode(tokenizer.encode(masked_text))
    # return new_text
    return masked_text
    # return str(text)


# 为每个文本生成新的句子
augmented_dataset = []
# 读取文本文件中的数据，然后扩充数据，得到augmented_dataset列表
data = []
with open('../data/testdata/all_data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        label, text = line.strip().split('\t')
        print(text)
        print(label)
        text = replace_with_mask(text)
        print(text)
        # 默认生成三个句子
        new_sentences = generate_sentences(text, label)
        # new_sentences = generate_sentences2(text, label)
        augmented_dataset.extend(new_sentences)
print(augmented_dataset)

# 将扩充的文本写进文件
add_sentence_txt = '../data/testdata/add_sentence.txt'
# add_sentence_txt = '../data/222.txt'
with open(add_sentence_txt, 'w') as f:
    f.seek(0)
    f.truncate()
with open(add_sentence_txt, 'a', encoding='utf-8') as f:
    for data in augmented_dataset:
        f.write(data)

# 读取未扩充前的数据到data2列表中
data_path = '../data/testdata/all_data.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    data2 = f.readlines()
print(data2)

# 将原数据列表和扩充数据列表连接成一个新的列表
all_data_list2 = augmented_dataset + data2
# 将文本打乱顺序
random.shuffle(all_data_list2)
# 返回随机打乱的数据列表
all_data_path2 = "../data/testdata/all_data2.txt"
# 清空文本的数据，并往all_data_path的txt文本中写入所有文本数据
with open(all_data_path2, 'w') as f:
    f.seek(0)
    f.truncate()
with open(all_data_path2, 'a', encoding='utf-8') as f:
    for data in all_data_list2:
        f.write(data)
print("数据文本生成成功！")
