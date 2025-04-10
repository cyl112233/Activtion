import torch
def compute(y_true, y_pred, num_classes):
    macro_Accuracy = (y_pred==y_true).sum()/y_true.shape[0]
    precision_list = []
    recall_list = []

    # 对每个类别计算精确率和召回率
    for c in range(num_classes):
        TP = torch.sum((y_true == c) & (y_pred == c)).float()
        FP = torch.sum((y_true != c) & (y_pred == c)).float()
        FN = torch.sum((y_true == c) & (y_pred != c)).float()

        precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
        recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)

        precision_list.append(precision)
        recall_list.append(recall)

    # 计算宏平均（Macro-average）
    macro_precision = torch.mean(torch.stack(precision_list))
    macro_recall = torch.mean(torch.stack(recall_list))



    return macro_precision, macro_recall,macro_Accuracy
