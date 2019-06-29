

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

如果考虑医学领域的特点，可细分为：
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

```python
#return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
return ["X", "B", "I", "E", "S", "O", "[CLS]", "[SEP]"]
```

**TODO**
非汉字字符串如何处理

## 模型

### CRF
1. 确定标签体系
BIO
BIESO
BIESO+
大部分情况下，标签体系越复杂准确度也越高，但相应的训练时间也会增加。因此需要根据实际情况选择合适的标签体系。


2. 确定特征模板文件
特征模版是一个文本文件，其中每行表示一个特征。CRF++会根据特征模版生成相关的特征函数。

```
#如下模板使用了unigram特征，并且仅以字符本身作为特征而不考虑其他特征。除当前字符外，还使用了其前后3个字，以及上下文的组合作为特征。
#Unigram
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U05:%x[-2,0]/%x[-1,0]/%x[0,0]
U06:%x[-1,0]/%x[0,0]/%x[1,0]
U07:%x[0,0]/%x[1,0]/%x[2,0]
U08:%x[-1,0]/%x[0,0]
U09:%x[0,0]/%x[1,0]

```

T**:%x[#,#]中的T表示模板类型，两个"#"分别表示相对的行偏移与列偏移。

一种是Unigram template:第一个字符是U，这是用于描述unigram feature（单字）的模板。

每一行 `%x[#,#]` 生成一个CRFs中的点(state)函数: `f(s, o)`, 其中s为t时刻的的标签(output)，o为t时刻的上下文.如CRF++说明文件中的示例函数:

func1 = if (output = B and feature="U02:那") return 1 else return 0

每一行模板生成一组状态特征函数，数量是L*N 个，L是标签状态数。N是此行模板在训练集上展开后的唯一样本数

它是由`U02:%x[0,0]`在输入文件的第一行生成的点函数.将输入文件的第一行"代入"到函数中,函数返回1;同时,如果输入文件的某一行在第1列也是“那”,并且它的output（第2列）同样也为B,那么这个函数在这一行也返回1。

另一种种是Bigram template（双字）：第一个字符是B，每一行 `%x[#,#]` 生成一个CRFs中的边(Edge)函数:`f(s', s, o)`, 其中`s'`为`t – 1`时刻的标签.也就是说,Bigram类型与Unigram大致机同,只是还要考虑到`t – 1`时刻的标签.如果只写一个B的话,默认生成`f(s', s)`，这意味着前一个output token和current token将组合成bigram features。


3. 处理数据文件
CRF模型的训练数据是一行一个token，一各序列（一句话）由多行token组成。
```
白 B
血 I
病 E
与 O
微 B
量 I
元 I
素 E
```


4. 运行模型
在训练命令中，template_file是模板文件，train_file是训练语料，都需要事先准备好；model是CRF++根据模板和训练语料生成的文件，用于解码。
在测试命令中，model_file是刚才生成的model文件，test_file是待测试语料，“>result_file”是重定向语句，指将屏幕输出直接输出到文件result_file中。
```shell
# train
crf_learn <template_file> <train_file> <model_file>
# test
crf_test -m <model_file> <test_file> > result_file
```



*扩展1： 结合规则进行改进*

同一实体内不同字间的类型不同，则以字类数较多者为
实体开头的字必定为B-???格式
实体的开始和结尾都有特定的特征可以遵循（如停用词、动词等作为分界等）
固定实体后跟实体应为B-???格式（如省名后）
实体间间隔较小时可能合并为同一实体

*扩展2： 结合分词及词性标注进行改进*

看来单从字的角度着眼已然不够，于是试图利用分词和词性标注信息。利用工具对文本进行分词标注，每一行的token仍然是以单字为特征，中间加入词性的信息如下图所示。针对这样的信息构建新的模板，利用中间一列的信息，可以提高准确率。


### Bi-LSTM-CRF
使用预训练字向量,作为embedding层输入,然后经过两个双向LSTM层进行编码,编码后加入dense层,最后送入CRF层进行序列标注

### BERT
使用预训练的Bert中文模型作为embedding层

how to run
```shell
# 人民日报语料
python bert_ner.py --data_dir=data/ChinaDaily/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/ChinaDaily/

# 白血病语料
python bert_ner.py --data_dir=data/bxb/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/bxb/

# 添加blstm-crf层
python bert_blstm_crf_ner.py --data_dir=data/bxb/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=data/vocab.txt --output_dir=./output/bert_blstm_crf/
```