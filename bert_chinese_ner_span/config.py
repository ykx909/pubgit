#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/01/23 13:05:22
@Author  :   Mr LaoChen
@Version :   1.0
'''

import torch
import argparse
from addict import Dict

class ArgsParse:

    @staticmethod
    def parse():
        # 命令行参数解析器
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # 模型训练参数
        parser.add_argument('--epochs', default=10, help='模型训练迭代数')
        parser.add_argument('--learn_rate', default=1e-5, help='学习率')
        parser.add_argument('--batch_size', default=32, help='训练样本批次数量')
        parser.add_argument('--mid_linear_dims', default=128, help='隐藏层大小')
        parser.add_argument('--dropout', default=0.1, help='dropout层比率')
        parser.add_argument('--per_steps_loss', default=100, help='模型训练计算平均loss的间隔')
        # 模型参数
        parser.add_argument('--bert_model', default='bert-base-chinese', help='bert模型名称')
        parser.add_argument('--local_model_dir', default='../bert_model/', help='bert模型本地缓存目录')
        parser.add_argument('--max_position_length', default=4096, help='模型输入position最大长度')
        # 语料参数
        parser.add_argument('--train_file', default='corpus/msra_mid/msra_train.json', help='训练语料文件')
        parser.add_argument('--test_file', default='corpus/msra_mid/msra_test.json', help='测试语料文件')
        # 模型保存参数
        parser.add_argument('--save_model_dir', default='./saved_model/', help='模型存盘文件夹')
        parser.add_argument('--load_model', default='ner_model_acc_99.27.pth', help='加载的模型存盘文件')

        return parser

    @staticmethod
    def extension(args):
        # 扩展参数
        options = Dict(args.__dict__)
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 实体标签
        options.tags = {"PER": 1, "ORG": 2, "LOC": 3}
        options.tags_rev = {1:"PER", 2:"ORG", 3:"LOC"}

        return options
		
    def get_parser(self):
        # 初始化参数解析器
        parser = self.parse()
        # 初始化参数
        parser = self.initialize(parser)
        # 解析命令行参数
        args = parser.parse_args()
        # 扩展参数
        options = self.extension(args)
        return options

def main():
    options = ArgsParse().get_parser()
    for opt in options:
        print(opt,":", options[opt])
    return options

if __name__ == '__main__':
    main()