import torch
from transformers import BertModel, BertConfig

# Bert模型结构配置参数对象
config = BertConfig()

# print(config)

# 通过配置文件创建Bert模型
model = BertModel(config)

# print(model)

# [CLS] == 101、[SEP] == 102
tokens = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102]]

# 转换tensor
model_inputs = torch.tensor(tokens)
# 模型推理
result = model(model_inputs)

# print(result['last_hidden_state'].shape)
# print(result['pooler_output'].shape)

# print(result.last_hidden_state.shape)
# print(result.pooler_output.shape)

# bert输出1: last_hidden_state 代表Bert Trm 最后的输出
print(result.last_hidden_state.shape)

# bert输出2: pooler_output 代表 Bert最后的输出+线性变换
print(result.pooler_output.shape)

print(result[0].shape)
print(result[1].shape)
