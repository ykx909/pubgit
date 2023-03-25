#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bert_span.py
@Time    :   2022/01/16 16:32:33
@Author  :   Mr LaoChen
@Version :   1.0
'''


import os
import torch
from tqdm import tqdm
import torchmetrics as metrics
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertSpan(nn.Module):

    def __init__(self, bert_model, mid_linear_dims, num_tags, dropout_prob=0.1):
        super(BertSpan, self).__init__()
        self.bert = bert_model
        self.num_tags = num_tags
        self.mid_linear_dims = mid_linear_dims
        self.dropout_prob = dropout_prob
        # Middle Linear
        self.mid_linear = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # Start Position
        self.start = nn.Linear(mid_linear_dims, num_tags)
        # End Position
        self.end = nn.Linear(mid_linear_dims, num_tags)

    def forward(self, 
                input_ids,
                token_type_ids,
                attention_mask,
                start_ids=None,
                end_ids=None):
        
        # bert模型输出(冻结参数)
        with torch.no_grad():
            bert_output = self.bert(input_ids,token_type_ids,attention_mask)
        # bert最后一层的hidden_state
        seq_out = bert_output.last_hidden_state[:,1:-1,:]
        # 中间层
        seq_out = self.mid_linear(seq_out)
        # 起始位置
        start_logits = self.start(seq_out)
        # 结束位置
        end_logits = self.end(seq_out)
        # 拼接logits
        out = (start_logits, end_logits)

        # 如果参数start_ids和end_ids不为None，则计算损失函数并同logits结果一并返回
        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.reshape(-1, self.num_tags)
            end_logits = end_logits.reshape(-1, self.num_tags)
            # 过滤 padding 部分，用于计算真实loss
            effect_loss = attention_mask[:,1:-1].reshape(-1) == 1  
            effect_start_logits = start_logits[effect_loss] 
            effect_end_logits = end_logits[effect_loss]
            # 真实标签采用同样的过滤，为和logits对齐
            effect_start_labels = start_ids.reshape(-1)[effect_loss]
            effect_end_labels = end_ids.reshape(-1)[effect_loss]
            # 交叉熵损失
            loss = nn.CrossEntropyLoss()
            start_loss = loss(effect_start_logits, effect_start_labels)
            end_loss = loss(effect_end_logits, effect_end_labels)
            loss = start_loss + end_loss
            # 拼接输出
            out = (loss,) + out
        return out