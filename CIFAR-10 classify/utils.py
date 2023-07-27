import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def plot_train_stats(train_acc):
    train_acc = torch.tensor(train_acc)
    train_acc_np = train_acc.cpu().detach().numpy()   # 将train_acc张量复制到CPU上并转换为NumPy数组
    fig, ax = plt.subplots()
    ax.plot(train_acc_np)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    plt.show()






