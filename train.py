import random
from utils import calculate_prf
from checkpoint import save_checkpoint
import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import swanlab
from torch.optim import AdamW



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
def evaluate_model(model, loader, dev,num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 保存全部样本真实标签、预测标签，用于后续计算P/R/F1
    all_true = []
    all_pred = []

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

            # 收集标签，传给utils计算P/R/F1
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())


    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples
    # 调用utils，计算P、R、F1
    precision, recall, f1 = calculate_prf(all_true, all_pred, num_classes)


    return avg_loss, acc,precision,recall,f1

def build_optimizer_and_scheduler(model, total_step, CFG):
    """构建分组优化器（带weight_decay）和学习率调度器"""
    no_decay = ["bias", "LayerNorm.weight"]
    group_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": CFG.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(group_params, lr=CFG.lr)
    warmup_step = int(total_step * CFG.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
    return optimizer, scheduler

def run_training_loop(model, train_loader, val_loader, optimizer, scheduler, num_label, CFG, dev):
    """训练主循环：epoch迭代、验证、早停、保存最优checkpoint"""
    best_val_acc = 0.0
    no_improve_count = 0
    patience = CFG.patience

    for epoch in range(CFG.epochs):
        print(f"\n======== Epoch {epoch + 1} / {CFG.epochs} ========")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, dev)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, dev, num_label)

        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "epoch": epoch + 1
        })
        print(f"train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_acc:{val_acc:.4f} | "
              f"val_precision:{val_precision:.4f} | val_recall:{val_recall:.4f} | val_f1:{val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(CFG.checkpoint_save_path, epoch, model, optimizer, scheduler, best_val_acc, no_improve_count)
            print(f"✅保存最优模型，best_val_acc = {best_val_acc:.4f}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"⚠️验证集无提升，no_improve_count={no_improve_count}/{patience}")

        if no_improve_count >= patience:
            print(f"\n🛑早停触发：连续{patience}轮验证集指标没有提升，终止训练！")
            break
    return best_val_acc