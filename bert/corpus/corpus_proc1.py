from datasets import load_dataset
import jieba
from tqdm import tqdm

# 加载语料
ds = load_dataset('amazon_reviews_multi','zh',split='train')

# 获取响应的字段
corpus = ds['review_body']

all_corpus = []
for c in tqdm(corpus):
    c = c.replace('。',' ').replace('！',' ').replace('？',' ').replace('，',' ')
    cs = [k for k in c.split() if k != '']
    # print(c)
    for j in range(len(cs)-1):
        s1 = ' '.join(jieba.lcut(cs[j]))
        s2 = ' '.join(jieba.lcut(cs[j+1]))
        all_corpus.append(s1 + '\t' + s2 + '\n')

# 另存语料文件
with open('bert_corpus.txt','w',encoding='utf-8') as f:
    f.writelines(all_corpus)

print('语料文件生成完毕!')
