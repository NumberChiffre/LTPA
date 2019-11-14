import os
import argparse
from time import gmtime, strftime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from LTPA.models.vgg_attention import VGGAttention
from LTPA.utils.utilities import *
from LTPA.utils.data_loader import get_data_loader
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
        self.params = vars(self).copy()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(
            log_dir=f'{ROOT_DIR}/{opt.logs_path}/tensorboard/'
                    f'{strftime("%Y-%m-%d", gmtime())}/'
                    f'{str(self.params)}_{strftime("%Y-%m-%d %H-%M-%S", gmtime())}')
        self.logger = ProjectLogger(level=loglevel)
        self.images = []
        self.image_dim = int(5 * 5)
        self.step = 0
        self.min_up_factor = 2

    def train_validate(self,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader):
        for epoch in range(self.opt.epochs):
            self.train(train_loader=train_loader, epoch=epoch)
            self.test(test_loader=test_loader, epoch=epoch)
        tb_layout = {
            'Training': {
                'Losses': ['Multiline',
                           ['epoch_train_loss', 'epoch_test_loss']],
                'Accuracy': ['Multiline',
                             ['epoch_train_acc', 'epoch_test_acc']],
            }
        }
        self.writer.add_custom_scalars(tb_layout)
        self.writer.close()

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              epoch: int):
        self.writer.add_scalar('train_learning_rate',
                               self.optimizer.param_groups[0]['lr'], epoch)
        self.logger.info(f'epoch {epoch} completed')
        self.scheduler.step()
        epoch_loss, epoch_acc = [], []
        for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
            inputs, targets = inputs.to(self.device), targets.to(
                self.device)
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            if batch_idx == 0:
                self.images.append(inputs[0:self.image_dim, :, :, :])
            pred, _, _, _ = self.model(inputs)
            loss = self.criterion(pred, targets)
            loss.backward()
            self.optimizer.step()
            predict = torch.argmax(pred, 1)
            total = targets.size(0)
            correct = torch.eq(predict, targets).cpu().sum().item()
            acc = correct / total
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"[epoch {epoch}][batch_idx {batch_idx}]"
                    f"loss: {round(loss.item(), 4)} "
                    f"accuracy: {100 * acc}% "
                )
            epoch_loss += [loss.item()]
            epoch_acc += [acc]
            self.step += 1
            self.writer.add_scalar('train_loss', loss.item(), self.step)
            self.writer.add_scalar('train_acc', acc, self.step)

        # log/add on tensorboard
        train_loss = np.mean(epoch_loss, axis=0)
        train_acc = np.mean(epoch_acc, axis=0)
        self.writer.add_scalar('epoch_train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch_train_acc', train_acc, epoch)
        self.logger.info(f"[epoch {epoch}] train_acc: {100 * train_acc}%")
        self.logger.info(f"[epoch {epoch}] train_loss: {train_loss}")
        self.inputs = inputs

        # save model params
        os.makedirs(f'{ROOT_DIR}/{opt.logs_path}/model_states', exist_ok=True)
        torch.save(self.model.state_dict(),
                   f'{ROOT_DIR}/{opt.logs_path}/model_states/net_epoch_{epoch}.pth')
        return train_loss, train_acc

    def test(self,
             test_loader: torch.utils.data.DataLoader,
             epoch: int):
        epoch_loss, epoch_acc = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader, 0):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                if batch_idx == 0:
                    self.images.append(
                        self.inputs[0:self.image_dim, :, :, :])
                pred, _, _, _ = self.model(inputs)
                loss = self.criterion(pred, targets)
                predict = torch.argmax(pred, 1)
                total = targets.size(0)
                correct = torch.eq(predict, targets).cpu().sum().item()
                acc = correct / total
                epoch_loss += [loss.item()]
                epoch_acc += [acc]

            # log/add on tensorboard
            test_loss = np.mean(epoch_loss, axis=0)
            test_acc = np.mean(epoch_acc, axis=0)
            self.writer.add_scalar('epoch_test_loss', test_loss, epoch)
            self.writer.add_scalar('epoch_test_acc', test_acc, epoch)
            self.logger.info(f"[epoch {epoch}] test_acc: {100 * test_acc}%")
            self.logger.info(f"[epoch {epoch}] test_loss: {test_loss}")

            # initial image..
            if epoch == 0:
                self.train_image = utils.make_grid(self.images[0],
                                                   nrow=int(
                                                       np.sqrt(self.image_dim)),
                                                   normalize=True,
                                                   scale_each=True)
                self.test_image = utils.make_grid(self.images[1],
                                                  nrow=int(
                                                      np.sqrt(self.image_dim)),
                                                  normalize=True,
                                                  scale_each=True)
                self.writer.add_image('train_image', self.train_image, epoch)
                self.writer.add_image('test_image', self.test_image, epoch)

            # training image sets
            __, ae1, ae2, ae3 = self.model(self.images[0])
            attn1 = plot_attention(self.train_image, ae1,
                                   up_factor=self.min_up_factor,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('train_attention_map_1', attn1,
                                  epoch)

            attn2 = plot_attention(self.train_image, ae2,
                                   up_factor=self.min_up_factor * 2,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('train_attention_map_2', attn2,
                                  epoch)

            attn3 = plot_attention(self.train_image, ae3,
                                   up_factor=self.min_up_factor * 4,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('train_attention_map_3', attn3,
                                  epoch)

            # validation image sets
            __, ae1, ae2, ae3 = self.model(self.images[1])
            attn1 = plot_attention(self.test_image, ae1,
                                   up_factor=self.min_up_factor,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('test_attention_map_1', attn1,
                                  epoch)
            attn2 = plot_attention(self.test_image, ae2,
                                   up_factor=self.min_up_factor * 2,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('test_attention_map_2', attn2,
                                  epoch)
            attn3 = plot_attention(self.test_image, ae3,
                                   up_factor=self.min_up_factor * 4,
                                   nrow=int(np.sqrt(self.image_dim)))
            self.writer.add_image('test_attention_map_3', attn3,
                                  epoch)
            return test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LTPA")
    parser.add_argument("--attention_mode", type=str, default="dp",
                        help="mode for running the attention model [pc or dp]")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--logs_path", type=str, default="logs",
                        help='path of log files')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count())
        print(f'Using {device}: {torch.get_num_threads()} threads')

    # load data
    train_loader, test_loader = get_data_loader(opt, im_size=32)

    # model + loss function + optimizer + scheduler
    net = VGGAttention(mode=opt.attention_mode)
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
