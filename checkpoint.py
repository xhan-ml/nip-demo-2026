import torch
import os

def save_checkpoint(save_path, epoch, model, optimizer, scheduler, best_val_acc, no_improve_count):
    # 新增：自动创建文件夹
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

