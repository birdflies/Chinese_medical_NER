

# medical NER

## 数据来源与预处理
### 数据来源
万方医学期刊论文题录数据（题名，关键词）

### 预定义实体
字符角色定义 BIESO
B，即Begin，表示开始
I，即Intermediate，表示中间
E，即End，表示结尾
S，即Single，表示单个字符
O，即Other，表示其他，用于标记无关字符
如果考虑医学领域的特点，可细分为
TREATMENT 治疗方式
BODY 身体部位
SIGNS 疾病症状
CHECK 医学检查
DISEASE 疾病实体
MEDICINE 药物实体

*在期刊论文数据中，大部分是DISEASE和MEDICINE类实体，可讨论是否加入实体类别特征对其识别效果的影响*

在`get_lables`里定义标签类别，如果标签类别的数量改变，需要修改相应的paddding数目

[X]：Mask掩盖，能更有效学习

[CLS]：每个序列的第一个 token 始终是特殊分类嵌入（special classification embedding），即 CLS。
对应于该 token 的最终隐藏状态（即，Transformer的输出）被用于分类任务的聚合序列表示。如果没有分类任务的话，这个向量是被忽略的。

[SEP]：用于分隔一对句子的特殊符号。有两种方法用于分隔句子：第一种是使用特殊符号 SEP；第二种是添加学习句子 A 嵌入到第一个句子的每个 token 中，
句子 B 嵌入到第二个句子的每个 token 中。如果是单个输入的话，就只使用句子 A 。(我们这里是单个输入)

作者：caoqi95
链接：https://www.jianshu.com/p/c59ae92a7a27
来源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
```python
#return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
return ["X", "B", "I", "E", "S", "O", "[CLS]", "[SEP]"]
```

**TODO**
每句话以空行分隔
非汉字字符串如何处理

## 模型

- Bi-LSTM-CRF
使用预训练字向量,作为embedding层输入,然后经过两个双向LSTM层进行编码,编码后加入dense层,最后送入CRF层进行序列标注.

how to run

- BERT_NER
使用预训练的Bert中文模型作为embedding层

how to run
```shell
# 人民日报语料
python bert_ner.py --data_dir=data/ChinaDaily/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/ChinaDaily/

# 白血病语料
python bert_ner.py --data_dir=data/bxb/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/bxb/

# 添加blstm-crf层
python bert_blstm_crf_ner.py --data_dir=data/bxb/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/bxb_crf/
```