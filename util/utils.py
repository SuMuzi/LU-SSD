import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
import json
import math
import random
import pywt
import pickle
import logging
import logging.handlers
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.cm import get_cmap
from thop import profile,clever_format
from thop.vision.basic_hooks import calculate_relu_flops
from scipy.fftpack import fftshift, fft2
# from thop import profile
from sklearn.metrics import accuracy_score,\
    confusion_matrix,precision_score,recall_score,f1_score,classification_report
from scipy.stats import pearsonr
# import cartopy.crs as ccrs
# import cartopy.feature as cfeat
class Weight_MAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_pred,y_true):
        weights = torch.clamp(torch.log(y_true + 1.1), np.log(0.1+1), np.log(100+1))

        return (weights * torch.abs(y_pred - y_true)).mean()

class High_freg_l1_loss(nn.Module):
    def __init__(self,alpha=1.0,eps=1e-8):
        super(High_freg_l1_loss, self).__init__()

        self.alpha = alpha
        self.eps = eps

    def forward(self,pred,target):
        lap = torch.abs(target-F.avg_pool2d(target,3,1,1)) #
        weight = torch.exp(self.alpha * lap)
        weight = weight/(weight.mean() + self.eps)
        loss = (weight * torch.abs(pred-target)).mean()

        return loss

def my_mse_weighted(y_pred, y_true):
    """
    y_pred/y_true: (B,1,156,192)
    """
    weights = torch.clamp(torch.log(y_true + 1.1), np.log(0.1+1), np.log(100+1))
    return (weights * (y_pred - y_true).abs()).mean()

def cal_accuracy(y_true,y_pred):
    acc = accuracy_score(y_true,y_pred)
    return acc

def cal_confusion_matrix(y_true,y_pred):
    c_m = confusion_matrix(y_true,y_pred)
    return c_m

def cal_precision_score(y_true,y_pred,average):
    p_s = precision_score(y_true,y_pred,average=average)
    return p_s

def cal_recall_score(y_true,y_pred,average):
    r_s = recall_score(y_true,y_pred,average=average)
    return r_s

def cal_f1_score(y_true,y_pred,average):
    f1 = f1_score(y_true,y_pred,average=average)
    return f1

def cal_classification_report(y_true,y_pred):
    c_r = classification_report(y_true,y_pred)
    return c_r

def cal_class_score(y_true,y_pred):

    # print(f'y_pred shape is {y_pred.shape}')
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # print(y_true.shape)
    # print(y_pred.shape)
    acc = accuracy_score(y_true,y_pred)
    c_m = confusion_matrix(y_true,y_pred)
    p_s = precision_score(y_true,y_pred,labels=[0,1,2,3,4],average=None)
    r_s = recall_score(y_true,y_pred,labels=[0,1,2,3,4],average=None)
    f1 = f1_score(y_true,y_pred,labels=[0,1,2,3,4],average=None)
    # c_r = classification_report(y_true,y_pred)

    return acc,c_m,p_s,r_s,f1


def print_parm_flops(model,data):
    flops, params = profile(model, inputs=(data,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"```````````````FLOPs: {flops}, Params: {params}  ````````````````````````")
    return flops, params

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer,is_sequential):

    if not is_sequential:

        assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                              'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
        if config.sch == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma,
                last_epoch=config.last_epoch
            )
        elif config.sch == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma,
                last_epoch=-1
            )
        elif config.sch == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.gamma,
                last_epoch=config.last_epoch
            )
        elif config.sch == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.eta_min,
                last_epoch=config.last_epoch
            )
        elif config.sch == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                threshold=config.threshold,
                threshold_mode=config.threshold_mode,
                cooldown=config.cooldown,
                min_lr=config.min_lr,
                eps=config.eps
            )
        elif config.sch == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                eta_min=config.eta_min,
                last_epoch=config.last_epoch
            )
        elif config.sch == 'WP_MultiStepLR':
            lr_func = lambda \
                epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
                [m for m in config.milestones if m <= epoch])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        elif config.sch == 'WP_CosineLR':
            lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                    math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    else:

        assert config.sch in ['CosineAnnealingLR'], 'Unsupported scheduler!'
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1,total_iters=config.warm_up_epochs)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min,
                last_epoch=config.last_epoch
            )

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          schedulers=[warmup,cosine],
                                                          milestones=[config.warm_up_epochs])
    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()
    


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        print('pred.size(0): ',pred.size(0))
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)

def SSIMLoss(input, target,C1 = 0.01 ** 2,C2=0.03 ** 2):

        mean_input = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        mean_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

        std_input = F.avg_pool2d(input - mean_input, kernel_size=3, stride=1, padding=1)
        std_target = F.avg_pool2d(target - mean_target, kernel_size=3, stride=1, padding=1)

        ssim_map = ((2 * mean_input * mean_target + C1) * (2 * std_input * std_target + C2)) / \
                   ((mean_input ** 2 + mean_target ** 2 + C1) * (std_input ** 2 + std_target ** 2 + C2))

        return -torch.mean(ssim_map)

def PSNRLoss(input, target,max_value=torch.tensor(1.0)):
    mse = F.mse_loss(input, target)
    psnr = 20. * (torch.log10(max_value) - torch.log10(mse))
    return -psnr

class L1PSNRSSIMLoss(nn.Module):
    def __init__(self, w1=100, w2=10, w3=1):
        super(L1PSNRSSIMLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, pred, target):
        ssimloss = SSIMLoss(pred, target)
        l1loss = self.l1(pred, target)
        psnrloss = PSNRLoss(pred, target)
        loss = self.w1 * ssimloss + self.w2 * l1loss + self.w3 * psnrloss
        return loss

def classify_loss(img,label):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(img,label)
    return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


def relu_flops(module,input,output=(1,36,36)):
    return output.numel()

def cal_params_flops(model, size, logger):
    input = torch.randn(1, size[0], size[1], size[2]).to('cuda')
    geo_lsm = torch.randn(1, 1, 288, 288).to('cuda')
    # custom_ops = {nn.ReLU:relu_flops}
    flops, params = profile(model, inputs=(input, geo_lsm))
    print('flops', flops / 1e9)
    print('params', params / 1e6)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: : {total / 1e6:.4f}')

def cal_params_flops_test(model, input,logger):
    # input = torch.randn(1, size[0], size[1], size[2]).to('cuda')
    # geo_lsm = torch.randn(1, 1, 288, 288).to('cuda')
    # custom_ops = {nn.ReLU:relu_flops}
    flops, params = profile(model, inputs=input)
    print('flops', flops / 1e9)
    print('params', params / 1e6)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: : {total / 1e6:.4f}')

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def reorder_image(img, input_order='HWC'):

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img
def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def statistics_class(prec_data):
    class_labels = np.zeros_like(prec_data, dtype=np.int64)
    class_labels = np.where((prec_data >= 0.0) & (prec_data < 0.1), 0, class_labels)
    class_labels = np.where((prec_data >= 0.1) & (prec_data < 2.5), 1, class_labels)
    class_labels = np.where((prec_data >= 2.5) & (prec_data < 10.0), 2, class_labels)
    class_labels = np.where((prec_data >= 10) & (prec_data < 50.0), 3, class_labels)
    class_labels = np.where((prec_data >= 50.0), 4, class_labels)
    return class_labels.astype(np.int64)

def Inverse_Log_Normlize_Precipitation(data, filepath):
    mean,std = get_mean_std(filepath)

    data_1 = data * std + mean
    data_2 = 10 ** data_1 - 1

    return data_2

def Inverse_Normlize_Precipitation_2(data,mean_tp,std_tp):

    data_1 = data * std_tp + mean_tp

    return data_1
def Inverse_Normlize_Precipitation(data,filepath):

    mean_tp,std_tp = get_mean_std(filepath)

    data_1 = data * std_tp + mean_tp

    return data_1
def Inverse_Log_Precipitation(data):

    data_initial = torch.pow(10,data) - 1

    return data_initial

def Inverse_MinMax_Precipitation(data,filepath):
    max_tp,min_tp = get_max_min(filepath)
    initial_data = data * (max_tp - min_tp) + min_tp
    return initial_data

def Inverse_Precipitation(data, filepath, mode):
    if mode == 'norm':
        prec = Inverse_Normlize_Precipitation(data,filepath)
    elif mode=='log':
        prec = Inverse_Log_Precipitation(data)
    elif mode == 'log_norm':
        prec = Inverse_Log_Normlize_Precipitation(data,filepath)

    elif mode == 'min_max':
        prec = Inverse_MinMax_Precipitation(data,filepath)
    else:
        print("Data pre-precess mode should be 'log','norm','log_norm', or 'min_max' !!!")
    return prec



def get_mean_std(means_std_path):
    with open(means_std_path, 'r', encoding='utf-8') as f:
        means_std = json.load(f)
    mean_tp = means_std['tp']['mean']
    std_tp = means_std['tp']['std']

    return mean_tp,std_tp
def get_mean_std_ifs(means_std_path):
    with open(means_std_path, 'r', encoding='utf-8') as f:
        means_std = json.load(f)
    mean_tp = means_std['mean']
    std_tp = means_std['std']

    return mean_tp,std_tp
def get_max_min(max_min_path):
    with open(max_min_path, 'r', encoding='utf-8') as f:
        means_std = json.load(f)
    max_tp = means_std['tp']['max']
    min_tp = means_std['tp']['min']
    return max_tp,min_tp

def check_data(data):
    valid_data = [0,1,2,3]
    has_invalid = np.any(~np.isin(data,valid_data))
    return has_invalid


def calculate_pearsonr(y_pred,y_true):
    corr, p = pearsonr(y_pred.ravel(), y_true.ravel())
    # print('Pearson correlation: %.3f' % corr)
    return corr,p

def prep_clf(y_pred, y_true, threshold):

    hits = np.sum((y_true >= threshold) & (y_pred >= threshold))

    false_alarms = np.sum((y_true < threshold) & (y_pred >= threshold))

    misses = np.sum((y_true >= threshold) & (y_pred < threshold))

    correctnegatives = np.sum((y_true < threshold) & (y_pred < threshold))

    return hits, misses, false_alarms, correctnegatives

def calculate_csi(y_pred, y_true):

    csis = []
    for threshold in [0.1,2.5,10.0,50.0]:
        hits, misses, false_alarms, correctnegatives = prep_clf(y_pred,y_true,threshold)
        csi = hits / (hits + false_alarms + misses)
        csis.append(csi)
    return csis


def calculate_precision(y_pred,y_true):

    precisions = []
    for threshold in [0.1,2.5,10.0,50.0]:
        TP, FN, FP, TN = prep_clf(y_pred,y_true, threshold)
        precision = TP / (TP + FP)
        precisions.append(precision)
    return precisions


def calculate_recall(y_pred,y_true):

    recalls = []
    for threshold in [0.1,2.5,10.0,50.0]:
        TP, FN, FP, TN = prep_clf(y_pred,y_true,threshold)
        rc = TP / (TP + FN)
        recalls.append(rc)
    return recalls


def calculate_ACC(y_pred,y_true):

    ACCs = []
    for threshold in [0.1,2.5,10.0,50.0]:
        TP, FN, FP, TN = prep_clf(y_pred,y_true, threshold)
        acc = (TP + TN) / (TP + TN + FP + FN)
        ACCs.append(acc)
    return ACCs
def calculate_BIAS(y_pred,y_true):


    Biass = []
    for threshold in [0.1, 2.5, 10.0,50.0]:
        hits, misses, falsealarms, correctnegatives = prep_clf(y_pred, y_true, threshold)
        bias=(hits + falsealarms) / (hits + misses)
        Biass.append(bias)
    return Biass


def calculate_fss(y_pred,y_true):

    FSSs = []

    thresholds = [0.1,2.5,10.0,50.0]
    windows_size=list(range(0, 43, 1))

    for threshold in thresholds:
        Fsss = []
        for window_size in windows_size:


            obs_binary = (y_true >= threshold).astype(np.int32)
            fcst_binary = (y_pred >= threshold).astype(np.int32)


            obs_pooled = pool2d(obs_binary, window_size)
            fcst_pooled = pool2d(fcst_binary, window_size)

            obs_prob = obs_pooled / window_size**2
            fcst_prob = fcst_pooled / window_size**2

            fbs = np.mean((fcst_prob - obs_prob) ** 2)

            fbs_max = np.mean((obs_prob - np.mean(obs_prob)) ** 2) * 2

            fss = 1 - fbs / fbs_max

            Fsss.append(fss)
        FSSs.append(Fsss)
    return FSSs

def pool2d(data, window_size):

    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    pooled_data = np.zeros_like(data, dtype=np.int32)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pooled_data[i, j] = np.sum(padded_data[i:i+window_size, j:j+window_size])

    return pooled_data


def get_sample_time(single_sample_index):
    start_time = '2018-01-01 00:00:00'
    time_series = pd.date_range(start=start_time, periods=40000, freq='H')

    time = str(time_series[single_sample_index])

    return time

def plt_discrete_indexs_img(indata,target,srdata,outpath):
    bounds_in = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    turbo_camp = get_cmap('turbo')
    colors = ['white']
    for i in range(9):
        colors.append(turbo_camp(i / 8))
    custom_cmap = ListedColormap(colors)
    norm_in = BoundaryNorm(bounds_in, custom_cmap.N)

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    im_in = ax1.imshow(indata, cmap=custom_cmap, norm=norm_in)
    cbar_in = fig.colorbar(im_in, label='Precipitation(mm/h)', ticks=bounds_in[:-1])
    cbar_in.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '50.0'])
    plt.title("Input data")

    ax2 = fig.add_subplot(1, 3, 2)
    im_out = ax2.imshow(target, cmap=custom_cmap, norm=norm_in)
    cbar_out = fig.colorbar(im_out, label='Precipitation(mm/h)', ticks=bounds_in[:-1])
    cbar_out.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '>50.0'])
    # plt.gca().invert_yaxis()
    plt.title("Target data")


    ax3 = fig.add_subplot(1, 3, 3)
    im_sr = ax3.imshow(srdata, cmap=custom_cmap, norm=norm_in)
    cbar_intpl = fig.colorbar(im_sr, label='Precipitation(mm/h)', ticks=bounds_in[:-1])
    cbar_intpl.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '>50.0'])
    # plt.gca().invert_yaxis()
    plt.title("SR data")

    plt.savefig(outpath)
    # print(f'save {outpath} successfully !!!')

    plt.close('all')

def plt_single_img(indata,outpath):
    bounds_in = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    turbo_camp = get_cmap('turbo')
    colors = ['white']
    for i in range(10):
        if i>0:
            colors.append(turbo_camp(i / 9))
    # colors = ['#009999','#66FF66','#329900','#EDB11F','#DF7D00','#FF3334','#FE0000','#990100']
    custom_cmap = ListedColormap(colors)
    norm_in = BoundaryNorm(bounds_in, custom_cmap.N)

    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(1, 1, 1)
    im_in = ax1.imshow(indata, cmap=custom_cmap, norm=norm_in)
    cbar_in = fig.colorbar(im_in, label='', ticks=bounds_in[:-1])
    cbar_in.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '50.0'])
    plt.title("LU-SSD Total Precipitation(mm/h)")

    plt.savefig(outpath)
    # print(f'save {outpath} successfully !!!')

    plt.close('all')


def plt_img(indata,outdata,outpath):

    bounds_in = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    camp_in = get_cmap('turbo', 10)
    norm_in = BoundaryNorm(bounds_in, camp_in.N)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    im_in = plt.imshow(indata, cmap=camp_in, norm=norm_in)
    cbar_in = plt.colorbar(im_in, label='Precipitation(mm/h)', ticks=bounds_in[:-1])
    cbar_in.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '50.0'])

    plt.title("Input data")

    bounds_out = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    camp_out = get_cmap('turbo', 10)
    norm_out = BoundaryNorm(bounds_out, camp_out.N)

    plt.subplot(1, 2, 2)
    im_out = plt.imshow(outdata, cmap=camp_out, norm=norm_out)
    cbar_out = plt.colorbar(im_out, label='Precipitation(mm/h)', ticks=bounds_out[:-1])
    cbar_out.set_ticklabels(['0.0', '0.1', '1.0', '2.5', '5.0', '10.0', '20.0', '30.0', '40.0', '50.0'])
    # plt.gca().invert_yaxis()
    plt.title("Target data")

    plt.savefig(outpath)
    print(f'save {outpath} successfully !!!')

    plt.close('all')


class StationaryWaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', level=1):
        super(StationaryWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        # Convert to numpy array
        x_np = x.detach().cpu().numpy()
        coeffs = pywt.swt(x_np, wavelet=self.wavelet, level=self.level)

        # Convert back to tensor
        coeffs_tensor = [torch.tensor(coeff, device=x.device) for coeff in coeffs]
        # print(len(coeffs_tensor))
        return coeffs_tensor


class InverseStationaryWaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', level=1):
        super(InverseStationaryWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, coeffs):
        # Convert to numpy array
        coeffs_np = [coeff.detach().cpu().numpy() for coeff in coeffs]
        x_reconstructed = pywt.iswt(coeffs_np, wavelet=self.wavelet)

        # Convert back to tensor
        x_reconstructed_tensor = torch.tensor(x_reconstructed, device=coeffs[0].device)
        return x_reconstructed_tensor

def save_npy_data(path,data):
    np.save(path,data)

class SWTLoss(nn.Module):
    def __init__(self, wavelet='haar', level=1):
        super(SWTLoss, self).__init__()
        self.swt = StationaryWaveletTransform(wavelet=wavelet, level=level)
        self.mse = nn.MSELoss()

    def forward(self, x, target):
        # Perform SWT on both input and target
        x_coeffs = self.swt(x)
        target_coeffs = self.swt(target)

        # Calculate loss for each level
        loss = 0
        for x_coeff, target_coeff in zip(x_coeffs, target_coeffs):
            x_coeff =  torch.tensor(x_coeff,requires_grad=True)
            target_coeff = torch.tensor(target_coeff,requires_grad=True)
            loss += self.mse(x_coeff, target_coeff)

        return loss

class WaveletHFLoss(nn.Module):

    def __init__(self, wt='haar', w=[1.0, 1.0, 1.0]):
        super().__init__()
        self.w = torch.tensor(w).view(1,1,3,1,1).cuda()
        self.wt = wt

    @torch.no_grad()
    def _wavelet(self, x):

        B, C, H, W = x.shape

        x_np = x.detach().cpu().numpy()          # (B,C,H,W)
        lh, hl, hh = [], [], []
        for bi in range(B):
            for ci in range(C):
                coeffs = pywt.dwt2(x_np[bi,ci], self.wt)
                LL, (LH, HL, HH) = coeffs

                LH = torch.from_numpy(pywt.idwt2((None,(LH,None,None)), self.wt)).unsqueeze(0)
                HL = torch.from_numpy(pywt.idwt2((None,(None,HL,None)), self.wt)).unsqueeze(0)
                HH = torch.from_numpy(pywt.idwt2((None,(None,None,HH)), self.wt)).unsqueeze(0)
                lh.append(LH); hl.append(HL); hh.append(HH)
        lh = torch.stack(lh).view(B,C,H,W).float().cuda()
        hl = torch.stack(hl).view(B,C,H,W).float().cuda()
        hh = torch.stack(hh).view(B,C,H,W).float().cuda()
        return lh, hl, hh

    def forward(self, sr, hr):
        lh_sr, hl_sr, hh_sr = self._wavelet(sr)
        lh_hr, hl_hr, hh_hr = self._wavelet(hr)
        loss = (self.w[0,0,0] * (lh_sr - lh_hr).pow(2) +
                self.w[0,0,1] * (hl_sr - hl_hr).pow(2) +
                self.w[0,0,2] * (hh_sr - hh_hr).pow(2)).mean()
        return loss
def calculate_rapsd_wavelength(field):
    field -= np.mean(field)

    window = np.hamming(field.shape[0])[:, None] * np.hamming(field.shape[1])[None, :]
    field *= window

    f_transform = fftshift(fft2(field))
    power = np.abs(f_transform) ** 2

    nx, ny = field.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0))
    kxx, kyy = np.meshgrid(kx, ky)
    k_radius = np.sqrt(kxx ** 2 + kyy ** 2)

    kbins = np.linspace(0, np.max(k_radius), min(nx, ny) // 2 + 1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    rapds = np.zeros_like(kvals)

    for i in range(len(kvals)):
        mask = (k_radius >= kbins[i]) & (k_radius < kbins[i + 1])
        if np.sum(mask) > 0:
            rapds[i] = np.mean(power[mask])
    rapds = 10 * np.log10(rapds)
    wavelengths = 2 * np.pi / kvals

    return wavelengths,rapds

def plot_rapsd(rapds,beam_names,outputpath):

    colors = ['red','black','blue','yellow','green','#FF5733','#7f7f7f','#8c564b','#9467bd','#bcbd22']
    spatial_distances = np.linspace(1 * 3, 144 * 3, 144)
    for i in range(rapds.shape[0]):

        plt.plot(spatial_distances, rapds[i], color=colors[i],label=f'{beam_names[i]}')

        plt.fill_between(spatial_distances,rapds[i]-5,rapds[i]+5,color=colors[i],alpha=0.3)

    plt.xlabel('Wavelenght (km)')
    plt.ylabel('RAPSD')

    plt.legend()

    plt.savefig(outputpath)


def plot_fss(fss,names,outputpath):

    colors = ['red','black','blue','yellow','green','#FF5733','#7f7f7f','#8c564b','#9467bd','#bcbd22']
    spatial_distances = np.linspace(0, 128, 3)
    for i in range(fss.shape[0]):

        plt.plot(spatial_distances, fss[i], color=colors[i],label=f'{names[i]}')

        plt.fill_between(spatial_distances,fss[i]-0.05,fss[i]+0.05,color=colors[i],alpha=0.3)

    plt.xlabel('Spatial scale (km)')
    plt.ylabel('FSS')

    plt.legend()

    plt.savefig(outputpath)

def calculate_mse(y_pred,y_true):
    mse = torch.mean((y_pred-y_true)**2)
    return mse

def calculate_rmse_mse(y_pred,y_true):
    mse = calculate_mse(y_true,y_pred)
    rmse = torch.sqrt(mse)

    return rmse,mse
def calculate_mae(y_pred,y_true):
    mae = torch.abs(y_pred-y_true)
    mae = torch.mean(mae)
    return mae

def save_results(data,out_path):
    with open(out_path,"w",encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)

    print(f"save {out_path} successfull !!!")

def load_results(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def correct_data(data):
    # data = np.clip(data,0,250)
    new_data = data.clip(0,100)
    return new_data


class GANLoss(nn.Module):


    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):

        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):

        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

def make_intensity_label(precip, K=5, method='fixed'):

    if method == 'quantile':
        bins = np.quantile(precip[precip > 0], np.linspace(0, 1, K+1))[1:-1]
    elif method == 'fixed':
        bins = [0.1, 2.5, 10, 50]
    return np.digitize(precip, bins).clip(0, K-1)

def update_bins(precip, K=5):

    valid = precip[precip > 0]
    if len(valid) == 0:
        return np.array([0.1, 1, 5, 10])

    bins = np.quantile(valid, np.linspace(0, 1, K+1))[1:-1]
    return bins.clip(0.1, None)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        ce = F.cross_entropy(logits, targets, reduction='none')  # -log(pt)
        pt = torch.exp(-ce)                                      # pt
        focal = (1 - pt) ** self.gamma * ce                      # (1-pt)^γ * -log(pt)
        if self.alpha is not None:
            alpha_t = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal = alpha_t * focal
        return focal.mean() if self.reduction == 'mean' else focal.sum()


class WeightScheduler:
    def __init__(self, alpha_start=1.0, alpha_end=0, initial_epoch=0,total_epochs=250, mode='linear'):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.in_epoch = initial_epoch
        self.total_epochs = total_epochs
        self.mode = mode
        self.max_epoch = self.in_epoch + self.total_epochs


    def get_alpha(self, epoch):
        if epoch < self.max_epoch:
            if self.mode == 'linear':
                return self.alpha_start - (self.alpha_start - self.alpha_end) * ((epoch-self.in_epoch) / self.total_epochs)
            elif self.mode == 'cosine':
                return self.alpha_end + (self.alpha_start - self.alpha_end) * (1 + np.cos(np.pi * (epoch-self.in_epoch) / self.total_epochs)) / 2
            elif self.mode == 'exp':
                gamma = 0.99
                return self.alpha_start * (gamma ** (epoch-self.in_epoch))
        else:
            return self.alpha_end

class RobustFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

        self.criterion_cls = nn.CrossEntropyLoss()
    def forward(self, logits, targets,alpha, gamma):

        B, K, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, K)   # (B*H*W, K)
        targets = targets.view(-1)                                       # (B*H*W,)

        ce = self.criterion_cls(logits, targets)          # -log(pt)
        pt = torch.exp(-ce)                                              # pt
        focal = (1 - pt) ** gamma * ce                      # (1-pt)^γ * -log(pt)  # (1-pt)^γ

        alpha_t = alpha[targets] if isinstance(alpha, torch.Tensor) else alpha
        focal = alpha_t * focal

        return focal.mean() if self.reduction == 'mean' else focal.sum()

class PixelFocalLoss(nn.Module):
    def __init__(self, bins, reduction='none'):
        super().__init__()
        self.criterion_cls = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        self.gamma = 2
        self.bins = bins

    def forward(self, logits, targets):

        B, K, H, W = logits.shape

        ce = self.criterion_cls(logits, targets)  # (B,H,W)
        pt = torch.exp(-ce)                                      # pt
        focal = (1 - pt) ** self.gamma                           # (1-pt)^γ

        freq = torch.bincount(targets.view(-1),minlength=K).float() / targets.numel()
        alpha = 1.0 / (freq + 1e-7)
        # print(f'alpha1 shape:{alpha.shape}')
        alpha = alpha[targets]                                   # (B,H,W)
        # print(f'alpha2 shape:{alpha.shape}')
        focal = alpha * focal

        return focal.mean() if self.reduction == 'mean' else focal.sum()

class PixelFocalLoss2(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.criterion_cls = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        self.gamma = 2

    def forward(self, logits, targets, alpha):

        B, K, H, W = logits.shape

        ce = self.criterion_cls(logits, targets)  # (B,H,W)
        pt = torch.exp(-ce)                                      # pt
        focal = (1 - pt) ** self.gamma                           # (1-pt)^γ

        # print(f'alpha1 shape:{alpha.shape}')
        alpha = alpha[targets]                                   # (B,H,W)
        # print(f'alpha2 shape:{alpha.shape}')
        focal = alpha * focal

        return focal.mean() if self.reduction == 'mean' else focal.sum()

def dynamic_alpha_gamma(epoch, total_epochs, class_counts, K):

    # freq = torch.bincount(class_counts.view(-1), minlength=K).float() / class_counts.numel()
    freq = class_counts/class_counts.numel()
    alpha = 1.0 / (freq + 1e-8)
    alpha = alpha / alpha.sum()
    # gamma = 2.0 + 1.0 * (1.0 - freq)
    # gamma = np.clip(gamma, 1.0, 4.0)

    alpha = alpha * (0.5 + 0.5 * torch.cos(torch.Tensor([torch.pi * epoch / total_epochs]).to(torch.device('cuda'))))

    return alpha

def correct_data2(data):
    min_val = np.amin(data,axis=(-2,-1),keepdims=True)
    new_data =  data - min_val
    new_data = np.clip(new_data, 0, 100)
    return new_data


class HaarWavelet2D(nn.Module):

    def __init__(self):
        super(HaarWavelet2D, self).__init__()


        h0 = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        h1 = torch.tensor([1.0, -1.0]) / np.sqrt(2)

        self.ll_filter = torch.outer(h0, h0).unsqueeze(0).unsqueeze(0)
        self.lh_filter = torch.outer(h0, h1).unsqueeze(0).unsqueeze(0)
        self.hl_filter = torch.outer(h1, h0).unsqueeze(0).unsqueeze(0)
        self.hh_filter = torch.outer(h1, h1).unsqueeze(0).unsqueeze(0)

        self.register_buffer('ll', self.ll_filter)
        self.register_buffer('lh', self.lh_filter)
        self.register_buffer('hl', self.hl_filter)
        self.register_buffer('hh', self.hh_filter)

    def forward(self, x):

        LL = F.conv2d(x, self.ll.repeat(x.size(1), 1, 1, 1), stride=2, padding=0, groups=x.size(1))
        LH = F.conv2d(x, self.lh.repeat(x.size(1), 1, 1, 1), stride=2, padding=0, groups=x.size(1))
        HL = F.conv2d(x, self.hl.repeat(x.size(1), 1, 1, 1), stride=2, padding=0, groups=x.size(1))
        HH = F.conv2d(x, self.hh.repeat(x.size(1), 1, 1, 1), stride=2, padding=0, groups=x.size(1))

        return LL, LH, HL, HH


class HaarInverseWavelet2D(nn.Module):

    def __init__(self):
        super(HaarInverseWavelet2D, self).__init__()

        h0 = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        h1 = torch.tensor([1.0, -1.0]) / np.sqrt(2)

        self.ll_recon = torch.outer(h0, h0).unsqueeze(0).unsqueeze(0)
        self.lh_recon = torch.outer(h0, h1).unsqueeze(0).unsqueeze(0)
        self.hl_recon = torch.outer(h1, h0).unsqueeze(0).unsqueeze(0)
        self.hh_recon = torch.outer(h1, h1).unsqueeze(0).unsqueeze(0)

        self.register_buffer('ll', self.ll_recon)
        self.register_buffer('lh', self.lh_recon)
        self.register_buffer('hl', self.hl_recon)
        self.register_buffer('hh', self.hh_recon)

    def forward(self, LL, LH, HL, HH):

        recon_ll = F.conv_transpose2d(LL, self.ll.repeat(LL.size(1), 1, 1, 1), stride=2, padding=0, groups=LL.size(1))
        recon_lh = F.conv_transpose2d(LH, self.lh.repeat(LH.size(1), 1, 1, 1), stride=2, padding=0, groups=LH.size(1))
        recon_hl = F.conv_transpose2d(HL, self.hl.repeat(HL.size(1), 1, 1, 1), stride=2, padding=0, groups=HL.size(1))
        recon_hh = F.conv_transpose2d(HH, self.hh.repeat(HH.size(1), 1, 1, 1), stride=2, padding=0, groups=HH.size(1))


        reconstructed = recon_ll + recon_lh + recon_hl + recon_hh

        return reconstructed


