import torch
import os

def save_checkpoint(save_path, epoch, model, optimizer, scheduler, best_val_acc, no_improve_count):
    """保存完整训练断点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "no_improve_count": no_improve_count
    }
    torch.save(checkpoint, save_path)
    print(f"✅ 保存完整断点 {save_path}")

def load_checkpoint(load_path, model, optimizer, scheduler, dev):
    """加载完整训练断点"""
    if not os.path.exists(load_path):
        print("⚠️ 无断点文件，从头训练")
        return 0, 0.0, 0
    checkpoint = torch.load(load_path, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    no_improve_count = checkpoint["no_improve_count"]
    print(f"✅ 加载断点，从epoch {start_epoch}继续训练")
    return start_epoch, best_val_acc, no_improve_count