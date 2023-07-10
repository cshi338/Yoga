import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import time
import sys
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be ")

from ResNet import Bottleneck, ResNet, ResNet50

best_prec1 = 0
resume = False #True if want to resume from saved model
resumePath = '' #Path of saved model
resumeStartEpoch = 0 #Epoch of resumed model
modelName = 'ResNet50'
num_classes = 10
epochs = 300
#Alter these parameters 
img_size = (100,100) #Size of resized input and output images
batch_size = 64

def main():
  global best_prec1, resume, batch_size, resumePath, resumeStartEpoch, modelName, img_size, num_classes, epochs
  transform_train = transforms.Compose([
      transforms.Resize(img_size),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  transform_test = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  kwargs = {'num_workers': 1, 'pin_memory': True}
  train_dataset = torchvision.datasets.ImageFolder('./Images/train/', transform=transform_train)
  val_dataset = torchvision.datasets.ImageFolder('./Images/test/', transform=transform_test)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
  val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

  assert torch.cuda.is_available()

  # Get the GPU device name.
  device_name = torch.cuda.get_device_name()
  n_gpu = torch.cuda.device_count()
  print(f"Found device: {device_name}, n_gpu: {n_gpu}")

  model = ResNet50(num_classes)
  model = model.cuda()

  # get the number of model parameters
  print('Number of model parameters: {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))
  # optionally resume from a checkpoint
  if resume == True:
      if os.path.isfile(resumePath):
          print("=> loading checkpoint '{}'".format(resumePath))
          checkpoint = torch.load(resumePath)
          resumeStartEpoch = checkpoint['epoch']
          best_prec1 = checkpoint['best_prec1']
          model.load_state_dict(checkpoint['state_dict'])
          print("=> loaded checkpoint '{}' (epoch {})"
                .format(resumePath, checkpoint['epoch']))
      else:
          print("=> no checkpoint found at '{}'".format(resumePath))

  cudnn.benchmark = True

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

  for epoch in range(epochs):
      # train for one epoch
      train(train_loader, model, criterion, optimizer, epoch)

      # evaluate on validation set
      prec1 = validate(val_loader, model, criterion, epoch)

      # remember best prec@1 and save checkpoint
      is_best = prec1 > best_prec1
      best_prec1 = max(prec1, best_prec1)
      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_prec1': best_prec1,
      }, is_best)
      print('Best accuracy: ', best_prec1.item())

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
          input_var = torch.autograd.Variable(input)
          target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(modelName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(modelName) + 'model_best.pth.tar')
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
