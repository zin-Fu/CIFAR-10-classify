from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from config import *

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机色彩变换
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'test': transforms.Compose([  # 测试集用的是真实世界的样本，不宜过度数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
}


train_dataset = datasets.CIFAR10(root='data',
                                 train=True,
                                 transform=data_transforms['train'],
                                 download=True)
# 只用训练集前30%来训练
train_size = int(0.3 * len(train_dataset))
train_indices = list(range(train_size))
train_sampler = SubsetRandomSampler(train_indices)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=8,
                          sampler=train_sampler)


test_dataset = datasets.CIFAR10(root='data',
                                train=False,
                                transform=data_transforms['test'])
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=8,
                         shuffle=False)

# 之后加上验证集


