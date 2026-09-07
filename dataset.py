import torch
from torch.utils.data import Dataset


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


def dynamic_collate_fn(batch):
    """动态padding，batch内部补齐到本批次最长长度"""
    input_ids_list = [item["input_ids"] for item in batch]
    attn_mask_list = [item["attention_mask"] for item in batch]
    label_list = [item["labels"] for item in batch]

    cur_max_len = max([len(x) for x in input_ids_list])

    batch_input_ids = []
    batch_attn_mask = []
    for ids, mask in zip(input_ids_list, attn_mask_list):
        pad_len = cur_max_len - len(ids)
        new_ids = ids + [0] * pad_len
        new_mask = mask + [0] * pad_len
        batch_input_ids.append(new_ids)
        batch_attn_mask.append(new_mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attn_mask, dtype=torch.long),
        "labels": torch.tensor(label_list, dtype=torch.long)
    }


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
