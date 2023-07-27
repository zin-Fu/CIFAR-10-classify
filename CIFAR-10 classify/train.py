import time
import torch.nn.functional as F
from utils import *
from dataloader import *
def train(model, optimizer):
    train_acc = []
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()
            optimizer.step()

            if not batch_idx % 20:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                      % (epoch + 1, NUM_EPOCHS, batch_idx,
                         len(train_loader), cost))
        model.eval()
        with torch.set_grad_enabled(False):  # 评估时节约内存
            train_acc.append(compute_accuracy(model, train_loader, device=DEVICE))
            print('Epoch: %03d/%03d | Train: %.3f%%' % (epoch + 1, NUM_EPOCHS, train_acc[-1]))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

    # 调用绘图函数
    plot_train_stats(train_acc)
    # 保存模型参数到文件中
    print("The training parameters have been saved to the file 'best.pt'😀 ")
    torch.save(model.state_dict(), "best.pt")

