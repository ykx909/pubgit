import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# 模型repo_id
checkpoint = "uer/roberta-base-finetuned-jd-full-chinese"
# 分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "店家发货很快，包装很严实，日期也很好，好评。", 
    "味道一般，不放冰箱很快就软化了。",
]

# padding批次语料自动长度填充
# truncation 超出模型最大输入长度512，自动截取
# return_tensors 返回值类型 np:ndarray pt:torch  tf:tensorflow张量
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# input_ids : token index 字符在字典中索引
# token_type_ids : 区分前后语句索引(默认第一句语句索引为0) [单一语句，这个可以不用提供]
# attention_mask : 批次语料中 padding填充准备mask （填充值为0） 

# 模型推理
# outputs = model(**inputs)
outputs = model(input_ids = inputs['input_ids'], 
                token_type_ids = inputs['token_type_ids'],
                attention_mask = inputs['attention_mask'])

# 分析结果
print(torch.argmax(outputs.logits, dim=-1))