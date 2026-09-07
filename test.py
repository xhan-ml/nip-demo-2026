from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CONFIG, DEVICE
from dataset import ToutiaoDataset
from train import evaluate_model


def run_test(model, test_text, test_label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    test_ds = ToutiaoDataset(test_text, test_label, tokenizer, CONFIG["max_len"], label2id)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loss, test_acc = evaluate_model(model, test_loader, DEVICE)
    print(f"\n========== 测试集评估 ==========")
    print(f"test_acc:{test_acc:.4f}, test_loss:{test_loss:.4f}")
    return test_loss, test_acc
