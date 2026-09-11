# main.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import swanlab
from checkpoint import save_checkpoint
from config import CFG
from model import BertTextClassifier
from train import set_seed, train_one_epoch, evaluate_model
from dataset import load_toutiao_single, ToutiaoDataset,dynamic_collate_fn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # 设置随机种子，保证结果可以复现
    set_seed(CFG.seed)

    # swanlab，使用CFG.to_dict()转字典给swanlab
    swanlab.init(
        project=CFG.swanlab_project,
        config=CFG.to_dict()
    )

    # 读取训练、验证、测试数据集（保留你自己的三个文件）
    train_texts, train_labels = load_toutiao_single(CFG.train_file)
    val_texts, val_labels = load_toutiao_single(CFG.val_file)
    test_texts, test_labels = load_toutiao_single(CFG.test_file)

    # 构造标签和数字的映射
    label_set = set(train_labels)
    label_list = sorted(list(label_set))
    label2id = {}
    for index, lab in enumerate(label_list):
        label2id[lab] = index

    print("训练集样本数量：", len(train_texts))
    print("验证集样本数量：", len(val_texts))
    print("测试集样本数量：", len(test_texts))
    print("分类类别数目：", len(label2id))
    print("label2id = ", label2id)

    # 加载bert分词器
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # 构建数据集对象
    train_dataset = ToutiaoDataset(train_texts, train_labels, tokenizer, CFG.max_len, label2id)
    val_dataset = ToutiaoDataset(val_texts, val_labels, tokenizer, CFG.max_len, label2id)
    test_dataset = ToutiaoDataset(test_texts, test_labels, tokenizer, CFG.max_len, label2id)

    # 构造DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              collate_fn=dynamic_collate_fn
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            collate_fn=dynamic_collate_fn
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=dynamic_collate_fn
                             )

    # 创建bert模型
    num_label = len(label2id)
    model = BertTextClassifier(
        model_name=CFG.model_name,
        num_classes=num_label,
        dropout_prob=CFG.dropout,
        hidden_size=CFG.hidden_size
    )
    model = model.to(dev)

    # 计算总步数、warmup步数
    total_step = len(train_loader) * CFG.epochs
    warmup_step = int(total_step * CFG.warmup_ratio)

    # ========== 优化器（BERT标准分组 + weight_decay） ==========
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": CFG.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_step
    )
    best_val_acc = 0.0
    patience = CFG.patience
    no_improve_count = 0
    # 训练循环
    for epoch in range(CFG.epochs):
        print("\n======== Epoch", epoch + 1, "/", CFG.epochs, "========")

        # 训练一轮
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, dev)
        # 验证集评估
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, dev,num_label)

        # 记录到swanlab
        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "epoch": epoch + 1
        })

        print("train_loss:", round(train_loss, 4),
              "| val_loss:", round(val_loss, 4),
              "| val_acc:", round(val_acc, 4),
              "| val_precision:", round(val_precision, 4),
              "| val_recall:", round(val_recall, 4),
              "| val_f1:", round(val_f1, 4))

        # 如果当前验证集准确率是历史最好，保存模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(CFG.checkpoint_save_path,
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            best_val_acc,
                            no_improve_count
                            )
            print("✅保存最优模型，best_val_acc =", round(best_val_acc, 4))
            no_improve_count = 0  # 有提升，重置计数
        else:
            no_improve_count += 1  # 没有提升，计数+1
            print(f"⚠️验证集无提升，no_improve_count={no_improve_count}/{patience}")

        # ========== 早停触发逻辑 ==========
        if no_improve_count >= patience:
            print(f"\n🛑早停触发：连续{patience}轮验证集指标没有提升，终止训练！")
            break
    print("\n全部轮次训练结束，加载最优模型做测试集评估")

    # 读取保存好的最优权重
    checkpoint = torch.load(CFG.checkpoint_save_path, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # 在测试集上评估


    with torch.no_grad():
        test_loss, test_acc ,test_precision, test_recall, test_f1= evaluate_model(model, test_loader, dev,num_label)

    swanlab.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,

    })

    print("\n【最终测试结果】")
    print("test_loss =", round(test_loss, 4))
    print("test_acc  =", round(test_acc, 4))
    print("test_precision =", round(test_precision, 4))
    print("test_recall    =", round(test_recall, 4))
    print("test_f1        =", round(test_f1, 4))

    swanlab.finish()


if __name__ == "__main__":
    main()
