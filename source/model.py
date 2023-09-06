# -*- coding:utf-8 -*-
# coding:unicode_escape
# @Author: Lemon00
# @Time: 2023/8/7 10:56
# @File: encoder_decoder
import math
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.nn.functional import log_softmax, pad


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """

        :param encoder: 传入一个编码器，将字符输入序列(x1,....,xn)，映射到新序列（自注意力序列）（z1,....,zn）
        :param decoder: 传入一个解码器，将序列（z1,....,zn），解码为字符输出序列（y1,...,yn）
        :param src_embed: 将源语言的字符转换为向量表示
        :param tgt_embed: 将目标语言的字符转换为向量表示
        :param generator: 接受decoder的输出，生成单词
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        !!!调用类中构建的 encode 和 decode 而不是传入的encoder和decoder
        在 encode 方法中，调用了传入的 encoder 类，返回一个（输入序列经过注意力机制处理后）向量表示
        在 decode 方法中，调用了传入的 decoder 类，返回 TODO：不知道返回的什么
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        先调用 src_embed 将输入序列转换为向量表示，
        再调用 encoder 将向量表示的输入序列转换为注意力机制处理后的向量表示
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码器的解码过程，根据输入的memory、源语言mask、目标语言输入tgt以及目标语言mask进行解码操作。
        :param memory: 编码器的输出memory
        :param src_mask: 源语言mask，用于掩盖源语言序列中填充位置的注意力计算
        :param tgt: 目标语言输入序列
        :param tgt_mask: 目标语言mask，用于掩盖目标语言序列中填充位置的注意力计算
        :return: 解码器的输出结果
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model: 单个单词使用多少个维度来表示
        :param vocab: 词汇表的大小
        具体说，传入一个shape
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    创造一个N层的Encoder整体
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差链接
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """

        :param x: 原始输入
        :param sublayer: 需要调用的层
        :return:
        """
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器实现
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        `self.sublayer[0]` 是一个子层，它期望的第二个参数是一个函数，而不是函数的返回值。这样做的目的是为了将函数的计算延迟到子层的内部，这样可以更加灵活地控制计算的过程。
        使用 `lambda` 匿名函数语法可以将函数的计算延迟，并将其作为参数传递给 `self.sublayer[0]`。
        换句话说，`x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))` 的写法是为了确保
        `self.self_attn(x, x, x, mask)` 在 `self.sublayer[0]` 内部进行计算，而不是在调用 `self.sublayer[0]` 之前就计算好结果。
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):  # TODO: src_attn不是很懂
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    # d_k:维度（经过qkv矩阵计算后的结果的向量维度）
    d_k = query.size(-1)
    # 为了适应batch计算 ，query 的shape为 （batch_size,sequence_length,d_k）
    # 这里的transpose是为了K的转置对应公式，实际操作为将k的导数第二维
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 在位置1 插入一个维度，适应多头注意力结构
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 将输入维度d_model 转换为 h * d_k
        # 变量q, k, v 是结果向量，而参数 q, k, v是输入参数（在encoder中是三个相同的x，），通过三个线性层计算得到不同的qkv向量
        query, key, value = \
            [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for lin, x in zip(self.linears, (query, key, value))]

        # 计算attention分数
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # -1 代表的是单句词数，这一步将多头注意力拆分的注意力结果合并
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value

        # 多头注意力机制直接合并注意力效果并不好，增加了一层concat线性变换
        return self.linears[-1](x)


def test_multihead_attention():
    # 设置测试参数
    d_model = 16
    h = 2
    dropout = 0.1
    seq_len = 10
    batch_size = 4

    # 创建测试数据
    rand = torch.randn(batch_size, seq_len, d_model)
    query = rand
    key = rand
    value = rand
    mask = None

    # 初始化MultiHeadAttention模块
    multihead_attention = MultiHeadAttention(d_model, h, dropout)

    # 运行测试数据
    output = multihead_attention(query, key, value, mask)

    # 打印输出形状
    print('Output shape:', output.shape)

    # 打印注意力分数
    print('Attention scores:', multihead_attention.attn)


# 运行测试函数
# test_multihead_attention()


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model
#
#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


import matplotlib.pyplot as plt


def test_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    plt.figure(figsize=(8, 6))

    for dim in [4, 5, 6, 7]:
        plt.plot(list(range(100)), y[0, :, dim], label=f"Dimension {dim}")

    plt.xlabel("Position")
    plt.ylabel("Embedding")
    plt.legend()
    plt.show()


# test_positional()

def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    c = copy.deepcopy
    # 多头注意力机制
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    # 目标
    ys = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)

# def run_tests():
#     for _ in range(10):
#         inference_test()
#
# run_tests()
