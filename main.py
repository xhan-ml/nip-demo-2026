import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab
from utils import dynamic_collate_fn
from config import CFG
from model import BertTextClassifier
from train import set_seed,  evaluate_model,build_optimizer_and_scheduler, run_training_loop
from dataset import ToutiaoDataManager

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # 设置随机种子，保证结果可以复现
    set_seed(CFG.seed)

    # swanlab，使用CFG.to_dict()转字典给swanlab
    swanlab.init(
        project=CFG.swanlab_project,
        config=CFG.to_dict()
    )

    # 加载bert分词器
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    data_mgr = ToutiaoDataManager(CFG, tokenizer)
    data_mgr.print_info()
    num_label = len(data_mgr.label2id)

    # 构造DataLoader
    train_loader = DataLoader(data_mgr.train_ds,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              collate_fn=dynamic_collate_fn
                              )
    val_loader = DataLoader(data_mgr.val_ds,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            collate_fn=dynamic_collate_fn
                            )
    test_loader = DataLoader(data_mgr.test_ds,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=dynamic_collate_fn
                             )

    # 创建bert模型
    model = BertTextClassifier(
        model_name=CFG.model_name,
        num_classes=num_label,
        dropout_prob=CFG.dropout,
        hidden_size=CFG.hidden_size
    )
    model = model.to(dev)

    # 构建优化器、scheduler
    total_step = len(train_loader) * CFG.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_step, CFG)

    # 执行训练循环
    run_training_loop(model, train_loader, val_loader, optimizer, scheduler, num_label, CFG, dev)

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
