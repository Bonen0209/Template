import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        true_positive = 0
        pred_positive = 0
        true_positive += torch.sum((pred == 1) * (target == 1)).item()
        pred_positive += torch.sum(pred == 1).item()

    # Return 0 when denominator is zero
    if pred_positive == 0:
        return 0.0 
    else:
        return true_positive / pred_positive


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        true_positive = 0
        positive = 0
        true_positive += torch.sum((pred == 1) * (target == 1)).item()
        positive += torch.sum(target == 1).item()
    
    # Return 0 when denominator is zero
    if positive == 0:
        return 0.0
    else:
        return true_positive / positive


def f1_score(output, target):
    # Return 0 when denominator is zero
    if (precision(output, target) + recall(output, target)) == 0:
        return 0.0
    else:
        return 2 * precision(output, target) * recall(output, target) / (precision(output, target) + recall(output, target))
