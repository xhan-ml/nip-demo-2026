# config.py
import json
import torch


class Config:
    def __init__(self, json_path="./configs/config1.json"):
        # 从json文件加载配置
        with open(json_path, encoding="utf-8") as f:
            cfg_data = json.load(f)

        # 把json里面的参数赋值给类的属性
        self.seed = cfg_data["seed"]
        self.max_len = cfg_data["max_len"]
        self.batch_size = cfg_data["batch_size"]
        self.epochs = cfg_data["epochs"]
        self.lr = cfg_data["lr"]
        self.warmup_ratio = cfg_data["warmup_ratio"]
        self.dropout = cfg_data["dropout"]
        self.patience = cfg_data["patience"]
        self.weight_decay = cfg_data["weight_decay"]
        self.model_name = cfg_data["model_name"]
        self.swanlab_project = cfg_data["swanlab_project"]
        self.checkpoint_save_path = cfg_data["checkpoint_save_path"]
        self.hidden_size = cfg_data["hidden_size"]
        self.train_file = cfg_data["train_file"]
        self.val_file = cfg_data["val_file"]
        self.test_file = cfg_data["test_file"]

        # 设备，不属于json配置，代码内部判断
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self):
        """转成字典，给swanlab用，swanlab需要字典格式config"""
        return {
            "seed": self.seed,
            "max_len": self.max_len,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "dropout": self.dropout,
            "model_name": self.model_name,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "swanlab_project": self.swanlab_project,
            "checkpoint_save_path": self.checkpoint_save_path,
            "hidden_size": self.hidden_size,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file
        }


# 实例化全局配置对象，项目其他文件直接导入使用
CFG = Config()
