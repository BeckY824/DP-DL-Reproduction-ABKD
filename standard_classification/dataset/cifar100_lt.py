from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

def get_data_folder():
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR100LTInstance(datasets.CIFAR100):
    """
    CIFAR100 Long-Tailed Instance Dataset.
    Inherits from standard CIFAR100 but cuts the training data to follow a long-tailed distribution.
    """
    def __init__(self, root, imb_factor=0.01, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        # 只有训练集需要做长尾处理，测试集(train=False)必须保持平衡
        if train:
            self.img_num_list = self.get_img_num_per_cls(100, imb_factor)
            self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
        self.data = np.vstack(new_data)
        self.targets = new_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 完美兼容原作者 KD/ABKD 的设定，返回 img, target, index
        return img, target, index


def get_cifar100_lt_dataloaders(batch_size=128, num_workers=8, imb_factor=0.01, is_instance=True):
    """
    Returns train_loader (Long-tailed), test_loader (Balanced), and n_data
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 训练集加载 Long-Tailed 变体
    train_set = CIFAR100LTInstance(root=data_folder,
                                   imb_factor=imb_factor,
                                   download=True,
                                   train=True,
                                   transform=train_transform)
    n_data = len(train_set)
    
    # 注意：保持与原代码 drop_last=True 的一致性
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers, 
                              drop_last=True)

    # 测试集必须使用标准的、平衡的 CIFAR-100 进行公平评估
    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader