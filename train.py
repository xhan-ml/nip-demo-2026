import random


import numpy as np
import torch
from tqdm import tqdm


# 设置随机种子保证可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, opt, sch, dev):
    model.train()
    total_loss = 0.0
    loss_func = torch.nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="train"):
        opt.zero_grad()

        input_ids = batch["input_ids"].to(dev)
        attn_mask = batch["attention_mask"].to(dev)
        labels = batch["labels"].to(dev)

        # 删掉 token_type_ids，你的数据集没有这个字段
        logits = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )

        loss = loss_func(logits, labels)
        loss.backward()
        opt.step()
        sch.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# 手写准确率计算，不再使用evaluate库
def evaluate_model(model, loader, dev):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_func = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            # 把数据搬到设备
            input_ids = batch["input_ids"].to(dev)
            attn_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)

            # 前向，得到logits
            logits = model(input_ids, attn_mask)
            loss = loss_func(logits, labels)

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.shape[0]

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples
    return avg_loss, acc

