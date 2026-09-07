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
    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(dev)
        attn_mask = batch["attention_mask"].to(dev)
        labels = batch["labels"].to(dev)
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
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
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            input_ids = batch["input_ids"].to(dev)
            attn_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            total_loss += loss.item()
            preds = torch.argmax(out.logits, dim=-1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    acc = total_correct / total_samples
    avg_loss = total_loss / len(loader)
    return avg_loss, acc
