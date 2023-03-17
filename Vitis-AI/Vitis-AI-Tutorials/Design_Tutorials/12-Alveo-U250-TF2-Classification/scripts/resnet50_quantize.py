# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18,resnet50

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device('cuda'):
    print("Using GPU\n")
else:
    print("Using CPU\n")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="images",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="pt_resnet50",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set'
)
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--finetune', 
    dest='finetune',
    action='store_true',
    help='finetune model before calibration')
args, _ = parser.parse_known_args()

def load_data(train=True,
              data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='resnet18',
              **kwargs):

  #prepare data
  # random.seed(12345)
  traindir = data_dir 
  valdir = data_dir
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if model_name == 'inception_v3':
    size = 299
    resize = 299
  else:
    size = 224
    resize = 256
  if train:
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
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

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model, val_loader, loss_fn):

  model.eval()
  model = model.to(device)
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  total = 0
  Loss = 0
  for iteraction, (images, labels) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    images = images.to(device)
    labels = labels.to(device)
    #pdb.set_trace()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    Loss += loss.item()
    total += images.size(0)
    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))
  return top1.avg, top5.avg, Loss / total

def quantization(title='optimize',
                 model_name='', 
                 file_path='', 
                 quant_mode='calib',
                 finetune=False):
  batch_size = 4

  model = resnet50().to(device)
  model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 3, 224, 224])
  if quant_mode == 'float':
    quant_model = model
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), output_dir="pt_resnet50/vai_q_output")

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  loss_fn = torch.nn.CrossEntropyLoss().to(device)

  val_loader, _ = load_data(
      subset_len=args.subset_len,
      train=False,
      batch_size=batch_size,
      sample_method='random',
      data_dir=args.data_dir,
      model_name=model_name)

  # finetune before calibration or load finetuned parameter before test
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=1024,
          train=False,
          batch_size=batch_size,
          sample_method=None,
          data_dir=args.data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # record  modules float model accuracy
  # add modules float model accuracy here
  acc_org1 = 0.0
  acc_org5 = 0.0
  loss_org = 0.0

  #register_modification_hooks(model_gen, train=False)
  acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)

  # logging accuracy
  print('loss: %g' % (loss_gen))
  print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False,output_dir="pt_resnet50/vai_q_output")


if __name__ == '__main__':

  model_name = 'resnet50'
  file_path = os.path.join(args.model_dir, model_name + '.pth')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path,
      quant_mode=args.quant_mode,
      finetune=args.finetune)

  print("-------- End of {} test ".format(model_name))

