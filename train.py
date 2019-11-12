import os
import argparse
from time import gmtime, strftime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from LTPA.models.vgg_attention import VGGAttention
from LTPA.utils.utilities import *
from LTPA import ROOT_DIR


class AttentionNetwork:
    def __init__(self,
                 opt: argparse.ArgumentParser,
                 model: torch.nn.Module,
                 criterion: torch.nn,
                 optimizer: optim.Optimizer,
                 scheduler: lr_scheduler.LambdaLR,
                 device: torch.device,
                 loglevel: int = 20,
                 ):
        self.opt = opt
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(f'{ROOT_DIR}/{opt.logs_path}/tensorboard/'
                                    f'{strftime("%Y-%m-%d", gmtime())}')
        self.logger = ProjectLogger(level=loglevel)
        self.images_disp = []
        self.image_dim = int(5 * 5)

    def train_validate(self,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader):
        for epoch in range(self.opt.epochs):
            self.train(train_loader=train_loader, epoch=epoch)
            self.test(test_loader=test_loader, epoch=epoch)

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              epoch: int):
        self.writer.add_scalar('train [learning_rate]',
                               self.optimizer.param_groups[0]['lr'], epoch)
        self.logger.info(f'epoch {epoch} completed')
        self.scheduler.step()
        step = 0
        cum_accuracy = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
            inputs, targets = inputs.to(self.device), targets.to(
                self.device)
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            if batch_idx == 0:
                self.images_disp.append(inputs[0:self.image_dim, :, :, :])
            pred, _, _, _ = self.model(inputs)
            loss = self.criterion(pred, targets)
            loss.backward()
            self.optimizer.step()

            # evaluate with current training
            if batch_idx % 10 == 0:
                self.model.eval()
                pred, _, _, _ = self.model(inputs)
                predict = torch.argmax(pred, 1)
                total = targets.size(0)
                correct = torch.eq(predict, targets).cpu().sum().item()
                accuracy = correct / total
                cum_accuracy = 0.9 * cum_accuracy + 0.1 * accuracy
                self.writer.add_scalar('train [loss]', loss.item(),
                                       step)
                self.writer.add_scalar('train [accuracy]', accuracy,
                                       step)
                self.writer.add_scalar('train [cum_accuracy]',
                                       cum_accuracy, step)
                self.logger.info(
                    f"[epoch {epoch}][batch_idx {batch_idx}]"
                    f"loss: {round(loss.item(), 4)} "
                    f"accuracy: {100 * accuracy}% "
                    f"running avg accuracy: {100 * cum_accuracy}%")
            step += 1

        self.inputs = inputs
        torch.save(self.model.state_dict(),
                   f'{ROOT_DIR}/{opt.logs_path}/model_states/net.pth')

    def test(self,
             test_loader: torch.utils.data.DataLoader,
             epoch: int):
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader, 0):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                if batch_idx == 0:
                    self.images_disp.append(
                        self.inputs[0:self.image_dim, :, :, :])
                pred_test = self.model(inputs)
                predict = torch.argmax(pred_test, 1)
                total += targets.size(0)
                correct += torch.eq(predict, targets).cpu().sum().item()
            self.writer.add_scalar('test [accuracy]', correct / total, epoch)
            self.logger.info(f"[epoch {epoch}] accuracy on test data: "
                             f"{100 * correct / total}")

            if opt.save_images:
                train_image = utils.make_grid(self.images_disp[0],
                                              nrow=int(np.sqrt(self.image_dim)),
                                              normalize=True, scale_each=True)
                self.writer.add_image('train [image]', train_image, epoch)
                if epoch == 0:
                    test_image = utils.make_grid(self.images_disp[1],
                                                 nrow=int(
                                                     np.sqrt(self.image_dim)),
                                                 normalize=True,
                                                 scale_each=True)
                    self.writer.add_image('test [image]', test_image, epoch)
            self.logger.info(f'logged images')

            if opt.save_images:
                min_up_factor = 2

                # training image sets
                __, ae1, ae2, ae3 = self.model(self.images_disp[0])
                if ae1 is not None:
                    attn1 = plot_attention(train_image, ae1,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('train [attention_map_1]', attn1,
                                          epoch)
                if ae2 is not None:
                    attn2 = plot_attention(train_image, ae2,
                                           up_factor=min_up_factor * 2,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('train [attention_map_2]', attn2,
                                          epoch)
                if ae3 is not None:
                    attn3 = plot_attention(train_image, ae3,
                                           up_factor=min_up_factor * 4,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('train [attention_map_3]', attn3,
                                          epoch)

                # validation image sets
                __, ae1, ae2, ae3 = self.model(self.images_disp[1])
                if ae1 is not None:
                    attn1 = plot_attention(test_image, ae1,
                                           up_factor=min_up_factor,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('test [attention_map_1]', attn1,
                                          epoch)
                if ae2 is not None:
                    attn2 = plot_attention(test_image, ae2,
                                           up_factor=min_up_factor * 2,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('test [attention_map_2]', attn2,
                                          epoch)
                if ae3 is not None:
                    attn3 = plot_attention(test_image, ae3,
                                           up_factor=min_up_factor * 4,
                                           nrow=int(np.sqrt(self.image_dim)))
                    self.writer.add_image('test [attention_map_3]', attn3,
                                          epoch)
                self.logger.info('logged attention maps')


def main():
    # use cpu or cuda depending on machines
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count())
        print(f'Using {device}: {torch.get_num_threads()} threads')

    # load data
    # TODO: change normalization constants for diff datasets..
    im_size = 32
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

    # model + loss function + optimizer + scheduler
    net = VGGAttention(mode='dp')
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(net, device_ids=list(
            range(torch.cuda.device_count()))).to(device)
    else:
        model = net.to(device)
    criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lr_lambda=lambda epoch: np.power(0.5, int(
                                          epoch / 25)))

    # time to train/validate
    obj = AttentionNetwork(opt=opt,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           device=device)

    obj.train_validate(train_loader=train_loader, test_loader=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LTPA")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--logs_path", type=str, default="logs",
                        help='path of log files')
    parser.add_argument("--save_images", action='store_true',
                        help='log images and (is available) attention maps')
    opt = parser.parse_args()
    main()
