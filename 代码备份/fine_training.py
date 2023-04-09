# 对chinese-bert-wwm-ext预训练模型进行微调来做文本分类任务

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("chinese-bert-wwm-ext", num_labels=num_labels)


import torch.optim as optim
import torch.nn as nn

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 开始微调：使用准备好的数据集、模型、优化器和损失函数进行微调。在每个epoch结束时，可以使用验证集进行模型评估
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    # evaluate on validation set
    with torch.no_grad():
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        for batch in val_dataloader:
            inputs, labels = batch
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_loss /= len(val_dataloader

# 我们可以使用训练数据集和测试数据集进行微调和评估：
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 加载chinesebert的tokenizer和预训练模型
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext",
                                                           num_labels=len(label_list))

# 加载训练数据集和测试数据集
train_dataset = ...
eval_dataset = ...

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# 设置Trainer并开始训练
trainer = Trainer


