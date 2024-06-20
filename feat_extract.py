import os
import os.path as osp
import sys
import random
import math
import copy
import argparse
import numpy as np
import scipy.io as sio

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils

def data_load(args):   
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dset_loaders = {}

    if args.dset != 'ImageNet-Sketch':
        txt_train_src = open(args.s_tr_dset_path).readlines()
        txt_val_src = open(args.s_val_dset_path).readlines()
        train_src_set = utils.ObjectImage('', txt_train_src, test_transform)
        val_src_set = utils.ObjectImage('', txt_val_src, test_transform)
        dset_loaders["source"] = torch.utils.data.DataLoader(train_src_set, batch_size=args.batch_size*3,
            shuffle=False, num_workers=args.worker, drop_last=False)
        dset_loaders["val"] = torch.utils.data.DataLoader(val_src_set, batch_size=args.batch_size*3,
            shuffle=False, num_workers=args.worker, drop_last=False)

    txt_tgt = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    target_set = utils.ObjectImage('', txt_tgt, train_transform)
    test_set = utils.ObjectImage('', txt_test, test_transform)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def generate_feature_wrapper(loader, model, args):
    def gather_outputs(selected_loader):
        bottleneck_layer = None
        if args.method == 'DINE' or (args.method == 'SHOT' and args.dset != 'ImageNet-Sketch'):
            backbone, bottleneck_layer, classifier_layer = model[0], model[1], model[2]
        else:
            backbone, classifier_layer = model[0],model[1]
        with torch.no_grad():
            start_test = True
            iter_loader = iter(selected_loader)
            for i in range(len(selected_loader)):
                inputs, labels = iter_loader.next()
                inputs = inputs.cuda()
                if bottleneck_layer is None and args.method !='SHOT':
                    fc_features, logit = classifier_layer(backbone(inputs))
                elif bottleneck_layer is None and args.method == 'SHOT':
                    fc_features = backbone(inputs)
                    logit = classifier_layer(fc_features)
                else:
                    fc_features = bottleneck_layer(backbone(inputs))
                    logit = classifier_layer(fc_features)

                if start_test:
                    features_ = fc_features.float().cpu()
                    outputs_ = logit.float().cpu()
                    labels_ = labels
                    start_test = False
                else:
                    features_ = torch.cat((features_, fc_features.float().cpu()), 0)
                    outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                    labels_ = torch.cat((labels_, labels), 0)
            return features_, outputs_, labels_

    def save(loader, data_name):
        features, outputs, labels = gather_outputs(loader)
        np.save(args.output_dir + '/' + data_name + '_feature.npy', features)
        np.save(args.output_dir + '/' + data_name + '_output.npy', outputs)
        np.save(args.output_dir + '/' + data_name + '_label.npy', labels)

    if args.dset != 'ImageNet-Sketch':
        print("-----------------Saving:source_train-----------")
        save(loader["source"], 'src_train')
        print("-----------------Saving:source_val-----------")
        save(loader["val"], 'src_val')
    print("-----------------Saving:target-----------")
    save(loader["test"], 'tgt')


def train(args):

    ## pre-process
    dset_loaders = data_load(args)

    ## base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()
    elif args.net == 'resnet34':
        netG = utils.ResBase34().cuda()  
    
    netB = None
    if args.method == 'MCD':
        netF = utils.ClassifierMCD(class_num=args.class_num, feature_dim=netG.in_features, 
            bottleneck_dim=args.bottleneck_dim).cuda()
    elif args.method == 'SAFN':
        netF = utils.ClassifierSAFN(class_num=args.class_num, feature_dim=netG.in_features, 
            bottleneck_dim=args.bottleneck_dim).cuda()
    elif args.method == 'MDD':
        netF = utils.ClassifierMDD(class_num=args.class_num, feature_dim=netG.in_features, 
            bottleneck_dim=args.bottleneck_dim, width=args.width).cuda()
    elif args.method == 'DINE' or (args.method == 'SHOT' and args.dset != 'ImageNet-Sketch'):
        netB = utils.feat_bottleneck(type=args.classifier, feature_dim=netG.in_features, 
            bottleneck_dim=args.bottleneck_dim).cuda()
        netF = utils.feat_classifier(type=args.layer, class_num = args.class_num, 
            bottleneck_dim=args.bottleneck_dim).cuda()
    elif args.method == 'SHOT' and args.dset == 'ImageNet-Sketch':
        netF = utils.res_classifier(res_name=args.net).cuda()
    else:
        netF = utils.ResClassifier(class_num=args.class_num, feature_dim=netG.in_features, 
            bottleneck_dim=args.bottleneck_dim).cuda()

    if netB is None:
        if args.method == 'SHOT':
            g_path = args.output_dir + '/target_F_par_0.3.pt'
            f_path = args.output_dir + '/target_C_par_0.3.pt'
            netG.load_state_dict(torch.load(g_path))
            netF.load_state_dict(torch.load(f_path))
            base_network = nn.Sequential(netG, netF)
        else:
            ckpt_path = osp.join(args.output_dir, args.ckpt + ".pt")
            base_network = nn.Sequential(netG, netF)
            base_network.load_state_dict(torch.load(ckpt_path))
    else:
        if args.method == 'SHOT':
            g_path = args.output_dir + '/target_F_par_0.3.pt'
            b_path = args.output_dir + '/target_B_par_0.3.pt'
            f_path = args.output_dir + '/target_C_par_0.3.pt'
        else:
            g_path = args.output_dir + '/dine_F.pt'
            b_path = args.output_dir + '/dine_B.pt'
            f_path = args.output_dir + '/dine_C.pt'
        netG.load_state_dict(torch.load(g_path))
        netB.load_state_dict(torch.load(b_path))
        netF.load_state_dict(torch.load(f_path))
        base_network = nn.Sequential(netG, netB, netF)

    base_network.eval()

    generate_feature_wrapper(dset_loaders, base_network, args)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN',
    'DANNE', 'MCD', 'MDD', 'SAFN', 'PADA', 'SHOT', 'DINE'], help="use official code for DA model training")
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'bnm', 'mcc', 'ent', 'bsp', 'atdoc'], 
                        help="implement all target-oriented DA methods within the ATDOC codebase")

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default=None, help="path of trained DA model")
    parser.add_argument('--output_src', type=str, default=None, help="path of source-only model in source-free UDA")
    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=1024, help="the dimension of the bottleneck layer")
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'], help="the type of UDA")


    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"], help="model setting for source-free UDA")
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"], help="model setting for source-free UDA")
    parser.add_argument('--width', type=int, default=1024, help="architecture setting for mdd")

    parser.add_argument('--max_epoch', type=int, default=1)
    #parser.add_argument('--smooth', type=float, default=0.1)
    #parser.add_argument('--hyperpara', type=float, default=1.0)
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet34", "resnet50", "resnet101"], help="the backbone used")
    parser.add_argument('--dset', type=str, default='DomainNet126', choices=['ImageNet-Sketch', 'DomainNet126', 'VISDA-C', 'office', 'office-home'], 
                        help="the DA dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    args = parser.parse_args()

    args.output = args.output.strip()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        args.max_epoch = 1
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        args.max_epoch = 1
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.max_epoch = 1
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.max_epoch = 1
    if args.dset == 'ImageNet-Sketch':
        names = ['imagenet', 'sketch']
        args.class_num = 1000
        args.max_epoch = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.da == 'uda':
        args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_list.txt'
    elif args.da == 'pda':
        args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_25_list.txt'
    args.test_dset_path = args.t_dset_path

    if args.pl != 'none':
        args.output_dir = osp.join(args.output, str(SEED), args.pl, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
    elif args.method in {'SHOT', 'DINE'}:
        args.output_dir = osp.join(args.output, str(SEED), args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
        args.output_src = osp.join(args.output_src, str(SEED), args.dset, 
            names[args.s][0].upper())
    else:
        args.output_dir = osp.join(args.output, str(SEED), args.method, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())


    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if args.method in {'SHOT', 'DINE'}:
        args.s_tr_dset_path = args.output_src + '/src_train.txt'
        args.s_val_dset_path = args.output_src + '/src_val.txt'
    else:
        args.s_tr_dset_path = args.output_dir + '/src_train.txt'
        args.s_val_dset_path = args.output_dir + '/src_val.txt'

    if not osp.exists(args.output_dir):
        print('Output dir not found! \n')

    if args.method != 'srconly': 
        args.ckpt = args.method

    if args.pl != 'none':
        args.ckpt = args.pl

    utils.print_args(args)
    train(args)
