
import os
import torch
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam

from ner_dataset import NerIterableDataset
from model_utils import custom_local_bert_tokenizer
from ner_dataset import NerDataset
from process import generate_dataloader, entity_collect
from bert_span import BertSpan
from torchmetrics import Accuracy
from model_utils import custom_local_bert, save_ner_model, load_ner_model

from config import ArgsParse
import warnings

# 禁用UserWarning
warnings.filterwarnings("ignore")

def train(opt, model, train_loader, test_loader):

    # 模型优化器
    optimizer = BertAdam(model.parameters(), lr=opt.learn_rate)
    
    for epoch in range(opt.epochs):
        pbar = tqdm(train_loader)
        # 模型训练
        model.train()
        for batch_data in pbar:
            # 模型训练张量注册到device
            batch_data = { k:v.to(opt.device) for k,v in batch_data.items() }
            loss = model(**batch_data)[0]
            # 计算并更新梯度
            loss.backward()
            optimizer.step()
            model.zero_grad()

            pbar.set_description('Epoch: %d/%d average loss: %.5f' % (epoch+1, opt.epochs, loss.item()))
        # 每轮epochs后评价
        acc = evaluate(opt, model, test_loader)
        # 每轮epochs后保存模型
        save_ner_model(opt, model, acc)

def evaluate(opt, model, test_dl):
    metric = Accuracy()
    metric.to(opt.device)
    pbar = tqdm(test_dl)
    # 模型评价
    model.eval()
    # 预测匹配的实体计数
    entities_count = {k:0 for k in opt.tags}
    for batch_data in pbar:
        batch_data = { k:v.to(opt.device) for k,v in batch_data.items()}
        # 提取到张量
        input_ids = batch_data['input_ids']
        token_type_ids = batch_data['token_type_ids']
        attention_mask = batch_data['attention_mask']
        start_ids = batch_data['start_ids']
        end_ids = batch_data['end_ids']
        # 模型推理
        start_logits, end_logits = model(input_ids,token_type_ids,attention_mask)
        mask = attention_mask[:,1:-1].bool()
        start_pred = torch.argmax(start_logits, dim=-1, keepdim=True)
        end_pred = torch.argmax(end_logits, dim=-1, keepdim=True)
        # 过滤掉padding
        start_pred = start_pred[mask]
        end_pred = end_pred[mask]
        start_ids = start_ids[mask]
        end_ids = end_ids[mask]
        # 计算准确率
        metric(start_pred, start_ids)
        metric(end_pred, end_ids)

        # 筛选非零内容
        start_idx = torch.nonzero(torch.squeeze(start_ids))
        start_pred = torch.squeeze(start_pred)[start_idx]
        end_idx = torch.nonzero(torch.squeeze(end_ids))
        end_pred = torch.squeeze(end_pred)[end_idx]
        # start和end完整预测的内容
        filted = (start_pred > 0) & (end_pred > 0)   
        real_pred = start_pred[filted]
        # 预测的实体
        tag_idx = [int(((i+1)/2).item()) for i in real_pred]
        for tidx in tag_idx:
            entities_count[opt.tags_rev[tidx]] += 1

        pbar.set_description('Accuracy')
    acc = metric.compute()
    print(entities_count)
    print(opt.entity_collect)
    print('Accuracy of the model on test: %.2f %%' %(acc.item()*100))
    return acc * 100

if __name__=='__main__':

    opt = ArgsParse().get_parser()

    # 加载本地缓存bert模型
    local = os.path.join(os.path.dirname(__file__),opt.local_model_dir, opt.bert_model)
    bert_model = custom_local_bert(local, max_position=opt.max_position_length)
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)
    
    # 训练用数据
    train_file = os.path.join(os.path.dirname(__file__), opt.train_file)
    test_file = os.path.join(os.path.dirname(__file__), opt.test_file)
    train_ds = NerIterableDataset(train_file)
    test_ds = NerDataset(test_file)
    train_dl = generate_dataloader(train_ds, tokenizer, opt.tags, opt.batch_size)
    test_dl = generate_dataloader(test_ds, tokenizer, opt.tags, 4)
    # 统计测试语料中实体数量
    opt.entity_collect = entity_collect(opt, test_ds)

    if len(opt.load_model) > 0:
        # 加载模型
        model = load_ner_model(opt)
    else:
        # 创建模型
        model = BertSpan(
            bert_model = bert_model,
            mid_linear_dims = opt.mid_linear_dims,
            num_tags=len(opt.tags)*2+1
        )
    model.to(opt.device)
    
    # 训练模型
    train(opt, model, train_dl, test_dl)


