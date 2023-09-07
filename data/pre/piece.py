# -*- coding:utf-8 -*- 
# coding:unicode_escape
# @Author: Lemon00
# @Time: 2023/8/12 14:37
# @File: piece
import sentencepiece as spm

# 训练
# spm.SentencePieceTrainer.train('--input=../zh_10M.csv --model_prefix=model-t-zh --vocab_size=65000')|
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
