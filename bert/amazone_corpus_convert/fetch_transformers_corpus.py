#!/usr/bin/env python
# -*- coding:utf-8
'''
@File	:	download_corpus.py
@Time	:	2022/02/11 12:04:43
@Author	:	Mr LaoChen
@Version:	1.0
'''

from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm
import os

def convert_corpus(dataloader, corpus):
    for item in tqdm.tqdm(dataloader):
        # 提取语料
        review = item['review_body'][0]
        # 按标点拆分子句
        subs = review.replace('，','·').replace('。','·').replace('！','·').replace('？','·').split('·')
        subs = [sub for sub in subs if sub != '']
        
        # 相邻子句组合为训练语料
        if len(subs) > 1:
            for i in range(len(subs) - 1):
                corpus.append(f'{subs[i]}\t{subs[i+1]}')

def save_corpus(corpus, corpu_file):
    with open(corpu_file, 'w', encoding='utf-8') as f:
        for corpu in tqdm.tqdm(corpus):
            f.write(corpu.strip() + '\n')

    print('语料文件', corpu_file, '保存成功!')

def fetch_corpus():
    # 动态下载amazone用户评论语料
    dataset = load_dataset("amazon_reviews_multi", "zh")

    print('训练集:', len(dataset['train']))
    print('验证集:', len(dataset['validation']))
    print('测试集:', len(dataset['test']))

    dl_train = DataLoader(dataset['train'])
    dl_valid = DataLoader(dataset['validation'])
    dl_test = DataLoader(dataset['test'])
    # 语料集合
    corpus = []
    convert_corpus(dl_train, corpus)
    convert_corpus(dl_valid, corpus)
    convert_corpus(dl_test, corpus)

    # 插入空格
    corpus = [''.join([c+' ' for c in corpu])[:-1] for corpu in corpus]
    # 语料存盘
    corpu_file = 'amazon_reviews.zh'
    save_corpus(corpus, corpu_file)

def main():
    corpu_file = 'amazon_reviews.zh'
    reader = open(corpu_file, 'r', encoding='utf-8')
    writer = open('_temp','w', encoding='utf-8')
    while True:
        line = reader.readline()
        if line.strip() == '': break
        
        if line[-1] != '\n':
            print(line, " -- 缺失换行\n")
            continue
        sentences = line.split('\t')
        if len(sentences) != 2:
            print(line, " -- 子句对长度不为2")
            continue
        elif sentences[0].strip() == '' or sentences[1].strip() == '':
            print(line, " -- 存在空子句")
            continue

        writer.write(line)

    reader.close()
    writer.close()

    # 替换原始语料文件为筛选后语料文件
    os.remove('amazon_reviews.zh')
    os.rename('_temp','amazon_reviews.zh')
    print('文件检查过滤完成!')

if __name__ == '__main__':
    main()
    


