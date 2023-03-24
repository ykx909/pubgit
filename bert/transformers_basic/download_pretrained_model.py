from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

# # bert模型配套分词器对象(自动加载vocab.txt)，实现token -> token_index
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# # 根据bert模型配置文件，加载还原模型结构和所有参数
# # ForMaskedLM代表给bert添加指定NLP任务头部
# model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

# print(model)


# 加载本地模型文件
tokenizer = AutoTokenizer.from_pretrained('models/roberta')
model = AutoModel.from_pretrained('models/roberta')

print(model)