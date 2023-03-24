from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# repo_id: model hub中唯一标志
# uer/roberta-base-finetuned-jd-full-chinese

# AutoTokenizer 分词器，包含Bert字典VOCAB
# 输入文本转换bert模型输入项(3个embeding的index)
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-full-chinese")

# 预训练模型结构(包含Bert + 推理层)
# ...ForSequenceClassification (针对序列进行分类)
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-full-chinese")

# 1. 输入文本tokenizer转换bert输入张量 “真是物美价廉” “商品包装都没拆，感觉效果可能一般”
input_text = input('请输入一段评论文本:')

# 2. 输入张量导入模型进行推理 (zero-shot) 0样本推理
# 任务名称: 'sentiment-analysis' 情感分析
text_classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
result = text_classification(input_text)

# 3. 分析模型输出结果
print(result)