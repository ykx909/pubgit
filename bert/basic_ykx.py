# <version 1406 1409 1418hub 1421-local 1424-yuebuqun-ee 1426-ykx 1718local>
import token
from unittest import result
from transformers import BertConfig,BertModel
import torch

config = BertConfig()
print(config)

model = BertModel(config)
print(model)

# encoded_sequences = torch.tensor(encoded_sequences)

# result = model(encoded_sequences)
print(result[0].shape)
print(result[1].shape)
print(result.last_hidden_state.shape)
print(result.pooler_output.shape)
print(result['last_hidden_state'].shape)
print(result['pooler_output'].shape)

from transformers import BertModel,AutoModel,AutoTokenizer,AutoModelForSequenceClassification,pipeline, AutoModelForMaskedLM
from huggingface_hub import snapshot_download
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="hfl/chinese-bert-wwm-ext", ignore_patterns=["*.msgpack", "*.h5"])
model = AutoModel()
while(1):
    input_text = input(">")
    if input_text =='q' or input_text == 'quit':
        break
    # input_data = tokenizer(input_text,return_tensors='pt',padding=True,truncation=True)
    # result = model(**input_data)
    # print(result)
    text_classification = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)
    result = text_classification(input_text)
    print(result)


model = BertModel.from_pretrained(r'C:/Users/cccql/.cache/huggingface/hub/models--hfl--chinese-bert-wwm-ext/')
print(model)
snapshot_download(repo_id="bert-base-chinese",ignore_regex=["*.msgpack", "*.h5"])


import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
 
model_name = 'bert-base-chinese'
MODEL_PATH = r'C:\Users\cccql\.cache\huggingface\hub\models--bert-base-chinese\snapshots\38fda776740d17609554e879e3ac7b9837bdb5ee'

# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name) 
# b. 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
print(bert_model)
