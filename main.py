# main.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import swanlab

from config import CFG
from model import BertTextClassifier
from train import set_seed, train_one_epoch, evaluate_model
from dataset import load_toutiao_single, ToutiaoDataset

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
    train_texts, train_labels = load_toutiao_single("./data/train_3k.txt")
    val_texts, val_labels = load_toutiao_single("./data/dev_1k.txt")
    test_texts, test_labels = load_toutiao_single("./data/test_1k.txt")

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
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

    # 创建bert模型
    num_label = 15
    model = BertTextClassifier(
        model_name=CFG.model_name,
        num_classes=num_label,
        dropout_prob=CFG.dropout
    )
    model = model.to(dev)

    # 计算总步数、warmup步数
    total_step = len(train_loader) * CFG.epochs
    warmup_step = int(total_step * CFG.warmup_ratio)

    # 优化器与学习率调度器
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_step
    )

    best_val_acc = 0.0

    # 训练循环
    for epoch in range(CFG.epochs):
        print("\n======== Epoch", epoch + 1, "/", CFG.epochs, "========")

        # 训练一轮
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, dev)
        # 验证集评估
        val_loss, val_acc = evaluate_model(model, val_loader, dev)

        # 记录到swanlab
        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1
        })

        print("train_loss:", round(train_loss, 4),
              "| val_loss:", round(val_loss, 4),
              "| val_acc:", round(val_acc, 4))

        # 如果当前验证集准确率是历史最好，保存模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./best_bert_toutiao.bin")
            print("✅保存最优模型，best_val_acc =", round(best_val_acc, 4))

    print("\n全部轮次训练结束，加载最优模型做测试集评估")
    model.eval()
    # 读取保存好的最优权重
    model.load_state_dict(torch.load("./best_bert_toutiao.bin", map_location=dev))
    # 在测试集上评估
    test_loss, test_acc = evaluate_model(model, test_loader, dev)

    swanlab.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })

    print("\n【最终测试结果】")
    print("test_loss =", round(test_loss, 4))
    print("test_acc  =", round(test_acc, 4))

    swanlab.finish()


if __name__ == "__main__":
    main()
