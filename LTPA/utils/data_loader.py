import os
import torchvision
import torchvision.transforms as transforms
import torch
from LTPA import ROOT_DIR


def get_data_loader(opt, im_size=32) -> torch.utils.data.DataLoader:
    transform_train = transforms.Compose([
        transforms.RandomCrop(im_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = torchvision.datasets.CIFAR10(root=f'{ROOT_DIR}/data/CIFAR10',
                                              train=True,
                                              download=True,
                                              transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=os.cpu_count())
    test_data = torchvision.datasets.CIFAR10(root=f'{ROOT_DIR}/data/CIFAR10',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=os.cpu_count())
    return train_loader, test_loader
