import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import evaluate



# ---------------------- 全局超参数配置 ----------------------
CONFIG = {
    "seed": 42,
    "max_len": 64,
    "batch_size": 2,
    "epochs": 6,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "model_name": "./model/bert-base-chinese",
    "swanlab_project": "bert-toutiao-classify"
}

# 设置随机种子保证可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import swanlab
import os
os.environ["SWANLAB_API_KEY"] = "CyiV2ZQHXK2jVUSnE3FGF"
swanlab.init(project=CONFIG["swanlab_project"])
swanlab.config.update(CONFIG)

# ---------------------- 读取已经切分好的三份数据集 ----------------------
def load_toutiao_single(filepath):
    texts = []
    labels = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")
            if len(parts) < 4:
                continue
            label_str = parts[1]
            text = parts[3]
            texts.append(text)
            labels.append(label_str)
    return texts, labels

train_text, train_label = load_toutiao_single("./data/train_3k.txt")
val_text, val_label = load_toutiao_single("./data/dev_1k.txt")
test_text, test_label = load_toutiao_single("./data/test_1k.txt")

# 构建标签映射，只用训练集标签
all_unique_labels = sorted(list(set(train_label)))
label2id = {lab: idx for idx, lab in enumerate(all_unique_labels)}
id2label = {idx: lab for idx, lab in enumerate(all_unique_labels)}

print(f"训练集:{len(train_text)}  验证集:{len(val_text)}  测试集:{len(test_text)}")
print(f"分类数目：{len(label2id)}")
print("label2id = ", label2id)

print("\n✅数据集读取测试完成！数据集部分没有问题。")
print("等你下载好bert‑base‑chinese模型文件之后，再取消下面一大段注释，跑完整训练。")




class ToutiaoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lab_str = self.labels[idx]
        lab = self.label2id[lab_str]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(lab, dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = BertForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=len(label2id)
).to(device)

train_ds = ToutiaoDataset(train_text, train_label, tokenizer, CONFIG["max_len"], label2id)
val_ds = ToutiaoDataset(val_text, val_label, tokenizer, CONFIG["max_len"], label2id)
test_ds = ToutiaoDataset(test_text, test_label, tokenizer, CONFIG["max_len"], label2id)

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

# Windows显式设置num_workers=0，单进程
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
total_steps = len(train_loader) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


print("已经执行完scheduler，准备加载metric")

print("已经执行完metric，准备进入循环训练")
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

best_val_acc = 0.0
for ep in range(CONFIG["epochs"]):
    print(f"\n===== Epoch {ep+1}/{CONFIG['epochs']} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    swanlab.log({"train_loss": train_loss,"val_loss": val_loss,"val_acc": val_acc,"epoch": ep+1})
    print(f"train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_acc:{val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./best_bert_toutiao.bin")
        print(f"保存最优模型, best_val_acc={best_val_acc:.4f}")

print("\n========== 测试集评估 ==========")
model.load_state_dict(torch.load("./best_bert_toutiao.bin", map_location=device))
test_loss, test_acc = evaluate_model(model, test_loader, device)
swanlab.log({"test_acc": test_acc, "test_loss": test_loss})
print(f"test_acc:{test_acc:.4f}")
swanlab.finish()

