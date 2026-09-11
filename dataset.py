import torch
from torch.utils.data import Dataset

#
# def load_toutiao_single(filepath):
#     texts = []
#     labels = []
#     with open(filepath, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split("_!_")
#             if len(parts) < 4:
#                 continue
#             label_str = parts[1]
#             text = parts[3]
#             texts.append(text)
#             labels.append(label_str)
#     return texts, labels


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
            truncation=True
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": lab
        }


# ===== 新增：数据管理类，封装加载/label2id/构造dataset =====
class ToutiaoDataManager:
    def __init__(self, cfg, tokenizer):
        # 加载三份数据
        self.train_texts, self.train_labels = self.load_toutiao_single(cfg.train_file)
        self.val_texts, self.val_labels = self.load_toutiao_single(cfg.val_file)
        self.test_texts, self.test_labels = self.load_toutiao_single(cfg.test_file)
        # 仅用训练集构建标签映射，避免数据泄露
        label_set = set(self.train_labels)
        label_list = sorted(label_set)
        self.label2id = {label: idx for idx, label in enumerate(label_list)}

        # 生成三个数据集实例
        self.train_ds = ToutiaoDataset(self.train_texts, self.train_labels, tokenizer, cfg.max_len, self.label2id)
        self.val_ds = ToutiaoDataset(self.val_texts, self.val_labels, tokenizer, cfg.max_len, self.label2id)
        self.test_ds = ToutiaoDataset(self.test_texts, self.test_labels, tokenizer, cfg.max_len, self.label2id)

    @staticmethod
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

    def print_info(self):
        print(f"训练集样本数量：{len(self.train_texts)}")
        print(f"验证集样本数量：{len(self.val_texts)}")
        print(f"测试集样本数量：{len(self.test_texts)}")
        print(f"分类类别数目：{len(self.label2id)}")
        print(f"label2id = {self.label2id}")
