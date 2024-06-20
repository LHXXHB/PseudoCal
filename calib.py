import os
import os.path as osp
import sys
import random
import math
import copy
import argparse
import csv
import numpy as np
import scipy.io as sio

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transcal_utils import get_weight, calculate_brier_score, ECELoss, VectorOrMatrixScaling, TempScaling, CPCS, TransCal, Oracle
import utils
import pdb


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

    txt_train_src = open(args.s_tr_dset_path).readlines()
    txt_val_src = open(args.s_val_dset_path).readlines()
    txt_tgt = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    
    train_src_set = utils.ObjectImage('', txt_train_src, test_transform)
    val_src_set = utils.ObjectImage('', txt_val_src, test_transform)
    target_set = utils.ObjectImage('', txt_tgt, train_transform)
    test_set = utils.ObjectImage('', txt_test, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(train_src_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)
    dset_loaders["val"] = torch.utils.data.DataLoader(val_src_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    dset_loaders["perm-source"] = torch.utils.data.DataLoader(train_src_set, batch_size=args.batch_size*3,
        shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["perm-val"] = torch.utils.data.DataLoader(val_src_set, batch_size=args.batch_size*3,
        shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["perm-test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc_error(logit, label):
    softmaxes = nn.Softmax(dim=1)(logit)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    accuracy = accuracies.float().mean()
    confidence = confidences.float().mean()
    error = 1 - accuracies.float()
    error = error.view(len(error), 1).float().numpy()
    return accuracy, confidence, error


## calibration methods: MatrixScaling, VectorScaling, TempScaling, CPCS, TransCal, PseudoCal, Oracle
def calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method=None, weight=None, bias_term=True, variance_term=True, args=None):
    ece_criterion = ECELoss()
    nll_criterion = nn.CrossEntropyLoss()
    if cal_method == 'VectorScaling' or cal_method == 'MatrixScaling':
        ece, nll, bs = VectorOrMatrixScaling(logits_source_val, labels_source_val, logits_target, labels_target, cal_method=cal_method)
        optimal_temp = 0.0
    else:
        if cal_method == 'TempScaling':
            ## directly use labeled source validation set for temperature scaling, corresponding to "TempScal-src" in the paper
            cal_model = TempScaling()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val)
        elif cal_method == 'CPCS':
            cal_model = CPCS()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val, torch.from_numpy(weight))
        elif cal_method == 'TransCal':
            cal_model = TempScaling()
            optimal_temp_source = cal_model.find_best_T(logits_source_val, labels_source_val)
            _, source_confidence, error_source_val = cal_acc_error(logits_source_val / optimal_temp_source, labels_source_val)

            cal_model = TransCal(bias_term, variance_term)
            optimal_temp = cal_model.find_best_T(logits_target.numpy(), weight, error_source_val, source_confidence.item())
        elif cal_method == 'Oracle':
            cal_model = Oracle()
            optimal_temp = cal_model.find_best_T(logits_target, labels_target)
        elif cal_method == 'pseudocal':
            soft_temp, hard_temp = pseudocal(args)
        if cal_method != 'pseudocal':
            ece = ece_criterion(logits_target / optimal_temp, labels_target).item()
            nll = nll_criterion(logits_target / optimal_temp, labels_target).item()
            bs = calculate_brier_score(logits_target / optimal_temp, labels_target).item()
    if cal_method != 'pseudocal':
        print(cal_method, round(ece, 4), round(nll, 4), round(bs, 4))
        return round(ece, 4), round(nll, 4), round(bs, 4), round(optimal_temp, 4)
    else:
        soft_ece = ece_criterion(logits_target / soft_temp, labels_target).item()
        soft_nll = nll_criterion(logits_target / soft_temp, labels_target).item()
        soft_bs = calculate_brier_score(logits_target / soft_temp, labels_target).item()
        hard_ece = ece_criterion(logits_target / hard_temp, labels_target).item()
        hard_nll = nll_criterion(logits_target / hard_temp, labels_target).item()
        hard_bs = calculate_brier_score(logits_target / hard_temp, labels_target).item()
        print(cal_method, round(soft_ece, 4), round(soft_nll, 4), round(soft_bs, 4), round(hard_ece, 4), round(hard_nll, 4), round(hard_bs, 4))
        return round(soft_ece, 4), round(soft_nll, 4), round(soft_bs, 4), round(soft_temp, 4), round(hard_ece, 4), round(hard_nll, 4), round(hard_bs, 4), round(hard_temp, 4)


def estimate_ece(args):
    task = args.task
    print(task)

    features_source_train = np.load(args.output_dir + '/' + 'src_train_feature.npy')
    logits_source_train = torch.from_numpy(np.load(args.output_dir + '/' + 'src_train_output.npy'))
    labels_source_train = torch.from_numpy(np.load(args.output_dir + '/' + 'src_train_label.npy')).long()

    features_target = np.load(args.output_dir + '/' + 'tgt_feature.npy')
    logits_target = torch.from_numpy(np.load(args.output_dir + '/' + 'tgt_output.npy'))
    labels_target = torch.from_numpy(np.load(args.output_dir + '/' + 'tgt_label.npy')).long()

    features_source_val = np.load(args.output_dir + '/' + 'src_val_feature.npy')
    logits_source_val = torch.from_numpy(np.load(args.output_dir + '/' + 'src_val_output.npy'))
    labels_source_val = torch.from_numpy(np.load(args.output_dir + '/' + 'src_val_label.npy')).long()

    accuracy_source_train, confidence_source_train, error_source_train = cal_acc_error(logits_source_train, labels_source_train)
    accuracy_target, confidence_target, error_target = cal_acc_error(logits_target, labels_target)
    accuracy_source_val, confidence_source_val, error_source_val = cal_acc_error(logits_source_val, labels_source_val)

    print("accuracy_source_train: {}".format(round(accuracy_source_train.item(), 4)))
    print("accuracy_source_val: {}".format(round(accuracy_source_val.item(), 4)))
    accuracy_target = round(accuracy_target.item(), 4)
    print("accuracy_target: {}".format(accuracy_target))

    ece_criterion = ECELoss()
    nll_criterion = nn.CrossEntropyLoss()

    src_tr_ece = round(ece_criterion(logits_source_train, labels_source_train).item(), 4)
    src_tr_nll = round(nll_criterion(logits_source_train, labels_source_train).item(), 4)
    src_tr_bs = round(calculate_brier_score(logits_source_train, labels_source_train).item(), 4)

    src_val_ece = round(ece_criterion(logits_source_val, labels_source_val).item(), 4)
    src_val_nll = round(nll_criterion(logits_source_val, labels_source_val).item(), 4)
    src_val_bs = round(calculate_brier_score(logits_source_val, labels_source_val).item(), 4)
    print("source_train_ece: {}".format(src_tr_ece))
    print("source_train_nll: {}".format(src_tr_nll))
    print("source_train_bs: {}".format(src_tr_bs))
    print("source_val_ece: {}".format(src_val_ece))
    print("source_val_nll: {}".format(src_val_nll))
    print("source_val_bs: {}".format(src_val_bs))

    ## Method 1: No Calibration (vanilla model without any calibration)
    tgt_ece = round(ece_criterion(logits_target, labels_target).item(), 4)
    tgt_nll = round(nll_criterion(logits_target, labels_target).item(), 4)
    tgt_bs = round(calculate_brier_score(logits_target, labels_target).item(), 4)
    print("vanilla_target_ece: {}".format(tgt_ece))
    print("vanilla_target_nll: {}".format(tgt_nll))
    print("vanilla_target_bs: {}".format(tgt_bs))


    repeat_times = 5
    for _idx_ in range(repeat_times):
        weight = get_weight(features_source_train, features_target, features_source_val)
        #pdb.set_trace()

        ## Method 2: Matrix Scaling (only use labeled source data)
        ece_ms, nll_ms, bs_ms, T_ms = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='MatrixScaling')

        ## Method 3: Vector Scaling (only use labeled source data)
        ece_vs, nll_vs, bs_vs, T_vs = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='VectorScaling')

        ## Method 4: Temperature Scaling (only use labeled source data)
        ece_ts, nll_ts, bs_ts, T_ts = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='TempScaling')

        ## Method 5: CPCS
        ece_CPCS, nll_CPCS, bs_CPCS, T_CPCS = calibration_in_DA(logits_source_val,labels_source_val, logits_target,labels_target, cal_method='CPCS', weight=weight)

        ## Method 6: TransCal
        ece_TC, nll_TC, bs_TC, T_TC = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='TransCal', weight=weight)

        ## Method 7: Oracle (assume labels in the target domain are aviable)
        ece_oracle, nll_oracle, bs_oracle, T_oracle = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='Oracle')

        ## Method 8: pseudocal (our method)
        ece_soft, nll_soft, bs_soft, T_soft, ece_hard, nll_hard, bs_hard, T_hard = calibration_in_DA(logits_source_val, labels_source_val, logits_target,
                        labels_target, cal_method='pseudocal', args=args)


        ece_result = [task, accuracy_target, src_tr_ece, src_val_ece, tgt_ece, ece_ms, ece_vs, ece_ts, ece_CPCS, ece_TC, ece_oracle, ece_soft, ece_hard]
        nll_result = [task, accuracy_target, src_tr_nll, src_val_nll, tgt_nll, nll_ms, nll_vs, nll_ts, nll_CPCS, nll_TC, nll_oracle, nll_soft, nll_hard]
        bs_result = [task, accuracy_target, src_tr_bs, src_val_bs, tgt_bs, bs_ms, bs_vs, bs_ts, bs_CPCS, bs_TC, bs_oracle, bs_soft, bs_hard]
        temp_result = [task, T_ms, T_vs, T_ts, T_CPCS, T_TC, T_oracle, T_soft, T_hard]


        ece_result = np.array(ece_result)
        with open(osp.join(args.save_dir, 'ece.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(ece_result)

        nll_result = np.array(nll_result)
        with open(osp.join(args.save_dir,'nll.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(nll_result)

        bs_result = np.array(bs_result)
        with open(osp.join(args.save_dir,'bs.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(bs_result)
        
        temp_result = np.array(temp_result)
        with open(osp.join(args.save_dir,'temp.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(temp_result)


def pseudocal(args):

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

    ## pseudo-target synthesis
    def mixup(select_loader):
        start_gather = True
        same_cnt = 0
        diff_cnt = 0
        total = 0
        all_diff_idx = None

        with torch.no_grad():
            for ep in range(args.max_epoch):
                for inputs, targets in select_loader:
                    batch_size = inputs.size(0)
                    sample_num = batch_size
                    inputs_a = inputs.cuda()
                    if args.aug == 'mixup':
                        clb_lam = args.lam
                    else:
                        clb_lam = 1
                        print("Not {}, only can use mixup!".format(args.aug))
                    rand_idx = torch.randperm(batch_size)
                    inputs_b = inputs_a[rand_idx]
                    outputs_a = base_network(inputs_a)
                    if type(outputs_a) is tuple:
                        soft_a = outputs_a[1]
                    else:
                        soft_a = outputs_a

                    soft_b = soft_a[rand_idx]
                    same_cnt += (soft_a.max(dim=-1)[1]==soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    diff_cnt += (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    
                    ## consider cross-cluster mixup to cover both correct and wrong predictions
                    diff_idx = (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0] + total

                    hard_a = F.one_hot(soft_a.max(dim=-1)[1], num_classes=args.class_num).float()
                    hard_b = hard_a[rand_idx]

                    mix_inputs = clb_lam * inputs_a + (1 - clb_lam) * inputs_b
                    mix_soft = clb_lam * soft_a.softmax(dim=-1) + (1 - clb_lam) * soft_b.softmax(dim=-1)
                    mix_hard = clb_lam * hard_a + (1 - clb_lam) * hard_b
                    mix_outputs = base_network(mix_inputs)
                    if type(mix_outputs) is tuple:
                        mix_out = mix_outputs[1]
                    else:
                        mix_out = mix_outputs

                    if start_gather:
                        all_mix_out = mix_out.detach().cpu()
                        all_mix_soft = mix_soft.detach().cpu()
                        all_mix_hard = mix_hard.detach().cpu()
                        all_diff_idx = diff_idx

                        start_gather = False
                    else:
                        all_mix_out = torch.cat((all_mix_out, mix_out.detach().cpu()), 0)
                        all_mix_soft = torch.cat((all_mix_soft, mix_soft.detach().cpu()), 0)
                        all_mix_hard = torch.cat((all_mix_hard, mix_hard.detach().cpu()), 0)
                        all_diff_idx = torch.cat((all_diff_idx, diff_idx), 0)


        mix_logits = all_mix_out[all_diff_idx]
        mix_soft_labels = all_mix_soft.max(dim=-1)[1][all_diff_idx]
        mix_hard_labels = all_mix_hard.max(dim=-1)[1][all_diff_idx]

        num_result = np.array([args.task, same_cnt, diff_cnt])
        with open(osp.join(args.save_dir,'mix_num.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(num_result)

        return mix_logits, mix_soft_labels, mix_hard_labels

    def ts(select_loader):
        mix_logits, mix_soft_labels, mix_hard_labels = mixup(select_loader)
        cal_model = TempScaling()
        soft_temp = cal_model.find_best_T(mix_logits, mix_soft_labels)
        hard_temp = cal_model.find_best_T(mix_logits, mix_hard_labels)
        return soft_temp, hard_temp

    soft_t, hard_t = ts(dset_loaders["perm-test"])

    return soft_t, hard_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Calibration in Domain Adaptation')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN',
                       'DANNE', 'MCD', 'MDD', 'SAFN', 'PADA', 'SHOT', 'DINE'], help="use official code for DA model training")
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'bnm', 'mcc', 'ent', 'bsp', 'atdoc'], 
                        help="implement all target-oriented DA methods within the ATDOC codebase")

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output_tr', type=str, default=None, help="path of trained DA model")
    parser.add_argument('--output_src', type=str, default=None, help="path of source-only model in source-free UDA")
    parser.add_argument('--save_dir', type=str, default='cal_logs', help="path of saved calibration results")
    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=1024, help="the dimension of the bottleneck layer")
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'], help="the type of UDA")

    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"], help="model setting for source-free UDA")
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"], help="model setting for source-free UDA")
    parser.add_argument('--width', type=int, default=1024, help="architecture setting for mdd")

    parser.add_argument('--alpha', type=float, default=1, 
                        help="the hyperparameter for beta distribution to generate random mix ratio used in training-stage mixup")
    parser.add_argument('--lam', type=float, default=0.65, help="our fixed mix ratio")
    parser.add_argument('--aug', type=str, default='mixup', choices=['trainaug', 'mocov2', 'randaug', 'cutmix', 'featmix', 'mixup'], 
                        help="strategy for synthesizing a labeled pseudo-target set")
    parser.add_argument('--max_epoch', type=int, default=1, help="epoch of inference-stage mixup to control the number of mixed samples")
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet34", "resnet50", "resnet101"], help="the backbone used")
    parser.add_argument('--dset', type=str, default='DomainNet126', choices=['ImageNet-Sketch', 'DomainNet126', 'VISDA-C', 'office', 'office-home'], 
                        help="the DA dataset used")

    args = parser.parse_args()

    args.output_tr = args.output_tr.strip()

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
        names = ['train', 'validation']
        args.class_num = 1000
        args.max_epoch = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #SEED = args.seed
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)
    #np.random.seed(SEED)
    #random.seed(SEED)

    if args.da == 'uda':
        args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_list.txt'
    elif args.da == 'pda':
        args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_25_list.txt'

    args.test_dset_path = args.t_dset_path
 
    if args.pl != 'none':
        args.output_dir = osp.join(args.output_tr, str(args.seed), args.pl, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
    elif args.method in {'SHOT', 'DINE'}:
        args.output_dir = osp.join(args.output_tr, str(args.seed), args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
        args.output_src = osp.join(args.output_src, str(args.seed), args.dset, 
            names[args.s][0].upper())
    else:
        args.output_dir = osp.join(args.output_tr, str(args.seed), args.method, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    args.save_dir = osp.join(args.save_dir, str(args.seed), 'calib_result')    
    if not osp.exists(args.save_dir):
        os.system('mkdir -p '+args.save_dir)

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

    if args.aug == 'mixup':
        args.task = args.da+'_'+args.ckpt+'_seed'+str(args.seed)+'_ep'+str(args.max_epoch)+'_'+args.aug+'_lam_'+str(args.lam)+'_'+args.dset+'_'+args.name
        
    utils.print_args(args)
    estimate_ece(args)
