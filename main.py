import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='~/data')
parser.add_argument('--log_name', type=str, default=None)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./checkpoints/resnet20_baseline')

# parser.add_argument('--Wbits', type=int, default=2)
# parser.add_argument('--Abits', type=int, default=2)
parser.add_argument('--arch', type=str, default='resnet18')


parser.add_argument('--bit', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=200)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

if cfg.log_name == None:
  cfg.log_name = "expr_w%sa%s" % (cfg.bit, cfg.bit)


cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'checkpoints', cfg.log_name)


os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu
cudnn.benchmark = True


def main():
  print('==> Preparing data ..')
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      # transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
  ])


  train_dataset = torchvision.datasets.CIFAR100(root=cfg.data_dir, train=True, download=True,
                          transform=transform_train)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

  eval_dataset = torchvision.datasets.CIFAR100(root=cfg.data_dir, train=False, download=True,
                         transform=transform_test)
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)

  print('==> Building ResNet..')
  if cfg.arch == "resnet18":
    from resnet import resnet18
    model = resnet18(bit=cfg.bit, num_classes=100).cuda()
  elif cfg.arch == "presnet20":
    from presnet import resnet20
    model = resnet20(wbits=cfg.bit, abits=cfg.bit, num_classes=100).cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)


  print ("Learnable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
  print (model); import time; time.sleep(5)
  if cfg.pretrain:
    model.load_state_dict(torch.load(cfg.pretrain_dir))

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    # pass
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))
    summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

  for epoch in range(cfg.max_epochs):
    train(epoch)
    test(epoch)
    lr_schedu.step(epoch)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()


if __name__ == '__main__':
  main()