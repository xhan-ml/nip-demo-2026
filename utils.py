import numpy as np

def calculate_prf(all_true, all_pred, num_classes):

    # ========== 1. 初始化混淆矩阵 ==========
    # 混淆矩阵 cm[真实标签][预测标签] = 样本数量
    # cm[k][j]：真实类别是k，预测成j类的样本个数
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_true, all_pred):
        cm[true_label][pred_label] += 1

    eps = 1e-8  # 极小值，防止分母等于0，出现除零报错

    p_list = []  # 保存每一类的Precision
    r_list = []  # 保存每一类的Recall
    f1_list = [] # 保存每一类的F1

    # ========== 2. 遍历每一个类别，单独计算P/R/F ==========
    for k in range(num_classes):
        # TP True Positive：真实是k类，预测也为k类（预测正确）
        tp = cm[k][k]
        # FP False Positive：真实不是k类，但是模型预测成k类（假阳性，误报）
        fp = np.sum(cm[:, k]) - tp
        # FN False Negative：真实是k类，但是模型预测成别的类别（假阴性，漏检）
        fn = np.sum(cm[k, :]) - tp

        # Precision 精确率：该类预测正确数量 ÷ 所有被预测为k类的样本总数
        precision_k = tp / (tp + fp + eps)
        # Recall 召回率：该类预测正确数量 ÷ 所有真实属于k类的样本总数
        recall_k = tp / (tp + fn + eps)
        # F1分数：P和R的调和平均，综合评价指标
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k + eps)

        p_list.append(precision_k)
        r_list.append(recall_k)
        f1_list.append(f1_k)

    # macro：所有类别指标取算术平均值，各类别权重一样
    macro_precision = np.mean(p_list)
    macro_recall = np.mean(r_list)
    macro_f1 = np.mean(f1_list)

    return macro_precision, macro_recall, macro_f1
