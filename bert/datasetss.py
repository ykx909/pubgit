from datasets import load_dataset
import jieba
from tqdm import tqdm
# ds = load_dataset('amazon_reviews_multi','zh',split='train')
# corpus = ds['review_body']
# all_corpus = []
# for c in tqdm(corpus):
#     c = c.replace('。',' ').replace('！',' ').replace('？',' ').replace('，',' ')
#     cs = c.split()
#     for j in range(len(cs) - 1):
#         s1 = ' '.join(jieba.lcut(cs[j]))
#         s2 = ' '.join(jieba.lcut(cs[j + 1])) 
#         all_corpus.append((s1 + '\t' + s2 + '\n'))

# with open('bert_corpus.txt','w',encoding='utf-8') as f:
#     f.writelines(all_corpus)

# print('语料生成完毕！')

ds = load_dataset('amazon_reviews_multi','zh',split='train')
corpus = ds['review_body']
all_corpus = []
for c in tqdm(corpus):
    c = c.replace('。',' ').replace('！',' ').replace('？',' ').replace('，',' ')
    cs = [k for k in c.split() if k != '']
    for j in range(len(cs) - 1):
        s1 = ' '.join(list(cs[j]))
        s2 = ' '.join(list(cs[j + 1])) 
        all_corpus.append((s1 + '\t' + s2 + '\n'))

with open('bert_corpus1.txt','w',encoding='utf-8') as f:
    f.writelines(all_corpus)

print('语料生成完毕！')

 
 

       
       