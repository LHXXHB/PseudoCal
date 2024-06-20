import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import math

from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.weight_norm as weightNorm


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

## network architecture
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class VGG16Base(nn.Module):
  def __init__(self):
    super(VGG16Base, self).__init__()
    model_vgg = torchvision.models.vgg16(pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)
    self.in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class ResBase34(nn.Module):
    def __init__(self):
        super(ResBase34, self).__init__()
        model_resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Block(nn.Module):
    def __init__(self, in_features, bottleneck_dim=1000, dropout_p=0.5):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x):
        f = self.fc(x)
        f = self.bn(f)
        f = self.relu(f)
        f = self.dropout(f)
        if self.training:
            f.mul_(math.sqrt(1 - self.dropout_p))
        return f

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

## the classifier used in imagenet pre-trained model
class res_classifier(nn.Module):
    def __init__(self, res_name):
        super(res_classifier, self).__init__()
        model_resnet = resnet_dict[res_name](pretrained=True)
        self.fc = model_resnet.fc
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.fc(x)
        return x        

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x,y

class ClassifierMDD(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=1024, width=1024):
        super(ClassifierMDD, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.bottleneck[1].weight.data.normal_(0, 0.005)
        self.bottleneck[1].bias.data.fill_(0.1)
        # The classifier head used for final predictions.
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num))
        for dep in range(2):
            self.fc[dep * 3].weight.data.normal_(0, 0.01)
            self.fc[dep * 3].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x,y
    
class ClassifierSAFN(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=1000, num_blocks=1, dropout_p=0.5):
        super(ClassifierSAFN, self).__init__()
        assert num_blocks >= 1
        layers = [nn.Sequential(
            Block(feature_dim, bottleneck_dim, dropout_p)
        )]
        for _ in range(num_blocks - 1):
            layers.append(Block(bottleneck_dim, bottleneck_dim, dropout_p))
        self.bottleneck = nn.Sequential(*layers)
        self.fc = nn.Linear(bottleneck_dim, class_num)

        for m in self.bottleneck.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x,y

class ClassifierMCD(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=1024):
        super(ClassifierMCD, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, class_num))

        for m in self.fc.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)

    def forward(self, x):
        y = self.fc(x)
        return x,y


class AdvFcMDD(nn.Module):
  def __init__(self, class_num, bottleneck_dim, width, max_iter=10000):
    super(AdvFcMDD, self).__init__()
    # The adversarial classifier head
    self.adv_fc = nn.Sequential(
                    nn.Linear(bottleneck_dim, width),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(width, class_num))
    for dep in range(2):
        self.adv_fc[dep * 3].weight.data.normal_(0, 0.01)
        self.adv_fc[dep * 3].bias.data.fill_(0.0)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    y = self.adv_fc(x)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


## data pre-processing
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    if isinstance(label, torch.utils.data.dataset.Subset) or isinstance(label, list):
        labeltxt = label
    else:
        labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if len(data) == 2:
            if is_image_file(data[0]):
                path = os.path.join(root, data[0])
                gt = int(data[1])
                item = (path, gt)
                # pdb.set_trace()
                images.append(item)
    return images

class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            #print(type(self.transform).__name__)
            if type(self.transform).__name__=='list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

def print_args(args):
    log_str = ("==========================================\n")
    log_str += ("==========       config      =============\n")
    log_str += ("==========================================\n")
    for arg, content in args.__dict__.items():
        log_str += ("{}:{}\n".format(arg, content))
    log_str += ("\n==========================================\n")
    print(log_str)
    #args.out_file.write(log_str+'\n')
    #args.out_file.flush()

## result calculation
def cal_fea(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels = iter_test.next()
            inputs = inputs.cuda()
            feas, outputs = model(inputs)
            if start_test:
                all_feas = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feas = torch.cat((all_feas, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feas, all_label

def cal_pred(loader, model, T):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels = iter_test.next()
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            sfmx_out = nn.Softmax(dim=1)(T * outputs)
            if start_test:
                all_sfmx = sfmx_out.cpu().float()
                start_test = False
            else:
                all_sfmx = torch.cat((all_sfmx, sfmx_out.cpu().float()), 0)
    return all_sfmx

def cal_acc(loader, model, flag=True, fc=None, nll=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    nll_test = 0
    if nll:
        nll_criterion = nn.CrossEntropyLoss()
        nll_test = nll_criterion(all_output, all_label.long()).item()

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if nll:
        return accuracy, predict, all_output, all_label, nll_test
    else:
        return accuracy, predict, all_output, all_label

def cal_acc_visda(loader, model, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean() / 100
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    print(acc)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return aacc, predict, all_output, all_label, acc
