from huggingface_hub import snapshot_download  # pip install huggingface_hub

# 借助hub模型本地缓存
snapshot_download(repo_id="hfl/chinese-roberta-wwm-ext", ignore_patterns=["*.msgpack", "*.h5"])


# from transformers import AutoModel

# # AutoModel只加载模型中Bert部分
model = AutoModel.from_pretrained(r'C:\Users\86135\.cache\huggingface\hub\models--hfl--chinese-bert-wwm-ext\snapshots\2a995a880017c60e4683869e817130d8af548486')

# print(model)
