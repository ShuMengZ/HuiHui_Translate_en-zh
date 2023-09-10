# 英-中通用翻译模型
以下记录英中翻译模型的开发记录和训练过程。
项目Github地址：https://github.com/ShuMengZ/HuiHui_Translate_en-zh
训练数据来自阿里： [WMT-Chinese-to-English-Machine-Translation-Training-Corpus](https://modelscope.cn/datasets/damo/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/summary)
计算平台：4090 * 2 数据并行策略
使用传统 Transformer 框架，引入 QuietAttention 和 RoPE 位置编码 提升模型性能，编码器和解码器的深度均为6，内部维度长512，参数规模37 M，训练消耗 24 GPU 时
## 模型构建
### Transformer 模型实现
使用Pytorh实现并改进了Transformer。
### RoPR位置编码
RoPE是一种在LLaMA、ChatGLM、Palm广泛引用位置编码格式，通过绝对位置编码的方式实现相对位置编码，作为替代Transformer的Sinusoidal位置编码，具有良好的外推性并具有远程衰减的性质，在本模型中使用是因为期望可以带来更强的长文本翻译性能。
### QuietAttention
作者EvanMiller7月24日发布于[博客](https://www.evanmiller.org/attention-is-off-by-one.html)，作者声称在Attention机制中的SoftMax函数实现存在数学原理上的错误，这个错误导致了在LLM训练中，会出现高出几个数量级的异常权重，使得训练过程中需要更多的内存来储存这些权重，浪费了珍贵的内存和计算性能。
作者提出修改SoftMax公式，在原有的分母上加一。这使得在x趋近于负无穷时，SoftMax的极限从1/k转换为0，这使得输出具有“均不相关“的数学可能。
![](images/new2.jpeg?v=1&type=image)

> 旧SoftMax

![](images/old.jpeg?v=1&type=image)

> 新SoftMax

![](images/new.jpeg?v=1&type=image)
在本模型中，本修改作用有限，仅是一次尝试。

## 模型训练流程
1. 训练tokenizer并构建词表——data/pre/piece.py	特殊词表建议与本项目相同。
```
spm.SentencePieceTrainer.train(  
    input=r'C:\project\WMT-Chinese-to-English-Machine-Translation-Training-Corpus\zh_24M.csv',  
  model_prefix='model-final-zh',  
  train_extremely_large_corpus=True,  
  vocab_size=65000,  
  bos_id=0,  
  pad_id=1,  
  eos_id=2,  
  unk_id=3,  
  )
```
2. 修改项目tokenizer——source/train.py 
```
def load_tokenizers():  
    try:  
        # 中文分词器  
		tokenizer_zh = spm.SentencePieceProcessor(model_file='../data/pre/model-final-24M-zh.model')  
    except IOError:  
        print("Tokenizer_zh can't load")  
    try:  
        # 英文分词器  
		tokenizer_en = spm.SentencePieceProcessor(model_file='../data/pre/model-final-24M-en.model')  
    except IOError:  
        print("Tokenizer_en can't load")  
    # 中文分词，英文分词  
	return tokenizer_zh, tokenizer_en
```

3. 修改数据地址——source/train.py
```
def create_dataloaders(......)
	......
	print("train data loading")  
	train_iter = mydataset('../data/t/train.csv')  
	# train_iter = mydataset('../data/t/24M-len64/train_len64.csv')  
	train_iter_map = to_map_style_dataset(train_iter)  
	train_sampler = (  
	    DistributedSampler(train_iter_map) if is_distributed else None  
	)  
	print("valid data loading")  
	valid_iter = mydataset('../data/t/val.csv')  
	# valid_iter = mydataset('../data/t/24M-len64/valid_len64.csv')  
	valid_iter_map = to_map_style_dataset(valid_iter)  
	valid_sampler = (  
	    DistributedSampler(valid_iter_map) if is_distributed else None  
)
```
4. 修改训练Config——source/train.py 适配你的训练卡
	如果你使用多卡训练请安装Nccl，并将Config中"distributed"设置为True。
```
def load_trained_model():
    config = {
        "batch_size": 128,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 64,
        "warmup": 3000,
        "fold_name": "",
        "file_prefix": "test_model"
    }
    # model_path = "test_model.pt"
    model_path = "test_model.pt"
    # if not exists(model_path):
    if True:
        folder_name = create_file(config)
        config["fold_name"] = folder_name
        config["file_prefix"] = "%s/%s" % (folder_name, config["file_prefix"])
        train_model(config)

    tokenizer_src, _ = load_tokenizers()
    model = make_model(len(tokenizer_src), len(tokenizer_src), N=6)
    model.load_state_dict(torch.load(model_path))
    return model
```
5. 模型训练——进入项目目录
	`python train.py`
6. 模型测试——source/train.py
```
# 测试用例  
def run_model_example(n_examples=5):  
    tokenizer_zh, tokenizer_en = load_tokenizers()  
  
    print("Preparing Data ...")  
    _, valid_dataloader = create_dataloaders(  
        torch.device("cpu"),  
  batch_size=1,  
  is_distributed=False,  
  )  
  
    print("Loading Trained Model ...")  
  
    model = make_model(len(tokenizer_zh), len(tokenizer_en), N=6)  
    model.load_state_dict(  
        torch.load("test_model-batch_size-128-num_epochs-8-2023-08-25 09-36/test_model00.pt",  
  map_location=torch.device("cpu"))  
    )  
  
    print("Checking Model Outputs:")  
    example_data = check_outputs(  
        valid_dataloader, model, tokenizer_zh, tokenizer_en, n_examples=n_examples  
    )  
    return model, example_data
```
7. 模型评分
	使用BeLU进行评分
9. 断点训练——source/train.py
	修改模型路径，修改train.py主函数，调用loading_config
```
def loading_config():
    print("Loading Tokenizer ...")
    tokenizer_zh, tokenizer_en = load_tokenizers()
    print("Loading Trained Model ...")
    model = make_model(len(tokenizer_zh), len(tokenizer_en), N=6)
    model.load_state_dict(
        torch.load("./test_model-batch_size-128-num_epochs-8-2023-08-25 09-36/test_model00.pt",
                   map_location=torch.device("cpu"))
    )
    config = {
        "batch_size": 128,
        "distributed": False,
        "num_epochs": 1,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 64,
        "warmup": 0,
        "fold_name": "",
        "file_prefix": "test_model"
    }
    folder_name = create_file(config)
    config["fold_name"] = folder_name
    config["file_prefix"] = "%s/%s" % (folder_name, config["file_prefix"])
    # 继续训练
    continue_training(0, 1, model, config)
```
10. 模型微调
	未完成 
### 训练代码优化
分布式训练带来的问题
1. 多线程加载数据，降低数据加载速度
2. 训练时减少CPU动作，减少GPU等待时间  
