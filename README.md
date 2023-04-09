项目名称：
ChineseBert_text_analysis_system
ZJF-Thunder/ChineseBert_text_analysis_system

大致使用说明：

1.系统运行必要文件：

需要去github仓库下载预训练模型chinese-bert-wwm-ext或bert-base-chinese或其他chinesebert模型，

保存到路径./models/chinese-bert-wwm-ext中

2.训练模型所需数据集存放在data目录中，

3.微调的模型存放路径为：./模型保存，

运行Text_Classification.py即可训练模型，并保存模型和测试模型,保存路径系统自动创建

4.Sentence_transformation.py为扩充数据集文件

5.其他py文件均为测试模型文件

6.后续再详细补充