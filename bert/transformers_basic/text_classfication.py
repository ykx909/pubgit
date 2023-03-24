from transformers import pipeline

# 流水线对象创建模型从输入到输出
classifier = pipeline(
    "sentiment-analysis",
    model="uer/roberta-base-finetuned-dianping-chinese")

output = classifier([
    "东西不错，份量手感，价格都没的说，点赞。关节阻尼合适。细节到位。多图，大家自己看。",
    "买了都没有拆，外包装就让人有种想退货的冲动"
])

print(output)