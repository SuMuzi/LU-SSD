import numpy as np
from tqdm import tqdm
import torch
import os
import yaml

from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from util.utils import save_imgs
from util.utils import calculate_psnr, calculate_ssim, tensor2np
import os
from util.utils import cal_class_score, statistics_class, calculate_csi
from util.utils import *
from torch.utils.tensorboard import SummaryWriter
def train_one_epoch(train_loader,
                    model,
                    criterion_reg,
                    criterion_cls,
                    criterion_wavelet,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    is_class,
                    weight_cls,
                    weight_reg,
                    weight_wavelet,
                    means_std_path,
                    writer,
                    scaler=None):
    '''
    train model for one epoch
    '''

    # switch to train mode
    train_total_step = len(train_loader)
    model.train()

    total_loss_list = []
    reg_loss_list = []
    cls_loss_list = []
    wavelet_loss_list = []

    # tp_mean,tp_std = get_mean_std(means_std_path)

    tq_train = tqdm(train_loader, desc=f'Training Epoch: {str(epoch)}', mininterval=0.3)
    wavelet_loss = 0
    loss_cls = 0
    for images, targets, geo_lsm, class_labels in tq_train:
        optimizer.zero_grad()

        images = images[:,0,:,:].unsqueeze(1).to('cuda')
        targets = targets[:,0,:,:].unsqueeze(1).to('cuda')
        # class_labels = class_labels.to('cuda', dtype=torch.long)
        # geo_lsm = geo_lsm.to('cuda')
        if config.amp:
            with autocast():
                if is_class:
                    out_reg, out_cls = model(images, geo_lsm)
                    # print(f'out_cls shape is: {out_cls.shape}')
                    # print(f'class_labels shape is: {class_labels.shape}')

                    # out_reg = out_reg * tp_std + tp_mean

                    loss_reg = criterion_reg(out_reg, targets)

                    loss_cls = criterion_cls(out_cls, class_labels)
                    loss_wavelet = criterion_wavelet(out_reg, targets)
                    loss = weight_cls * loss_cls + weight_reg * loss_reg + weight_wavelet * loss_wavelet
                else:
                    out_reg = model(images, geo_lsm)
                    # out_reg = out_reg * tp_std + tp_mean
                    loss_reg = criterion_reg(out_reg, targets)
                    loss_wavelet = criterion_wavelet(out_reg, targets)
                    loss = weight_reg * loss_reg + weight_wavelet * loss_wavelet

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:

            if is_class:
                out_reg, out_cls = model(images, geo_lsm)
                # print(f'targets shape is: {targets.shape}')
                # print(f'out_reg shape is: {out_reg.shape}')
                # out_reg = out_reg * tp_std + tp_mean
                loss_reg = criterion_reg(out_reg, targets)
                loss_cls = criterion_cls(out_cls, class_labels)
                loss_wavelet = criterion_wavelet(out_reg, targets)
                cls_loss_list.append(loss_cls.item())
                wavelet_loss_list.append(loss_wavelet.item())
                #loss = weight_cls * loss_cls + weight_reg * loss_reg + weight_wavelet * loss_wavelet
                loss = weight_cls * loss_cls + weight_reg * loss_reg
            else:
                out_reg = model(images, geo_lsm)
                # out_reg = out_reg * tp_std + tp_mean
                loss_reg = criterion_reg(out_reg, targets)
                # loss_wavelet = criterion_wavelet(out_reg, targets)
                # loss = weight_reg * loss_reg + weight_wavelet * loss_wavelet
                total_loss = weight_reg * loss_reg


            # print(loss)
            total_loss.backward()
            optimizer.step()

        total_loss_list.append(total_loss.item())
        reg_loss_list.append(loss_reg.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        tq_train.set_postfix({'loss': np.mean(total_loss.item())})
    tq_train.close()
    if is_class:
        mean_cls_loss = np.mean(cls_loss_list)
        mean_wavelet_loss = np.mean(wavelet_loss_list)
    else:
        mean_cls_loss = 0
        mean_wavelet_loss = 0
    mean_total_loss = np.mean(total_loss_list)
    mean_reg_loss = np.mean(reg_loss_list)

    writer.add_scalar('total_loss', mean_total_loss, epoch)
    writer.add_scalar('reg_loss', mean_reg_loss, epoch)
    writer.add_scalar('cls_loss', mean_cls_loss, epoch)
    writer.add_scalar('wavelet_loss', mean_wavelet_loss, epoch)

    log_info = f'[Training] epoch: {epoch}, loss: {mean_total_loss:.8f}, lr: {now_lr}, Max_Allocated_Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3)} GB'

    logger.info(log_info)
    print(log_info)
    scheduler.step()

    return model,mean_total_loss


def val_one_epoch(val_loader,
                  model,
                  epoch,
                  is_class,
                  indexs_path,
                  img_outpath,
                  logger,
                  writer,
                  mode='norm',
                  is_plt=False,
                  lr_means_std_path=None,
                  hr_means_std_path=None):
    # switch to evaluate mode
    indexs = np.load(indexs_path)
    # lr_tp_mean,lr_tp_std = get_mean_std(lr_means_std_path)
    # hr_tp_mean, hr_tp_std = get_mean_std(hr_means_std_path)

    val_total_step = len(val_loader)
    model.eval()
    psnr_scores = []
    ssim_scores = []
    acc_scores = []
    p_s_scores = []
    r_s_scores = []
    f1_scores = []
    c_r_scores = []
    c_m_scores = []
    csi_scores = []

    index_count = 0
    rmse_scores = []
    mae_scores = []
    mse_scores = []
    tq_valid = tqdm(val_loader, desc=f'Validation ', mininterval=0.3)
    for lr, hr, geo_lsm, class_labels in val_loader:#val_total_step
        with torch.no_grad():

            lr = lr[:,0,:,:].unsqueeze(1).to(torch.device('cuda'))
            hr = hr[:,0,:,:].unsqueeze(1).to(torch.device('cuda'))
            geo_lsm = geo_lsm.to('cuda')

            if is_class:
                sr, out_cls = model(lr, geo_lsm)
            else:
                sr = model(lr, geo_lsm)


            for j in range(hr.shape[0]):  # hr.shape[0]

                lr_img = lr[j,0,:,:]
                sr_img = sr[j,0,:,:]
                gt_img = hr[j,0,:,:]
                # Inverse Normlization

                lr_img_inverse = Inverse_Precipitation(lr_img, lr_means_std_path, mode)
                sr_img_inverse = Inverse_Precipitation(sr_img, hr_means_std_path, mode)
                gt_img_inverse = Inverse_Precipitation(gt_img, hr_means_std_path, mode)

                lr_img_inverse = correct_data(lr_img_inverse)
                sr_img_inverse = correct_data(sr_img_inverse)
                gt_img_inverse = correct_data(gt_img_inverse)

                rmse,mse = calculate_rmse_mse(sr_img_inverse, gt_img_inverse)
                mae = calculate_mae(sr_img_inverse, gt_img_inverse)

                rmse_scores.append(rmse.cpu().numpy())
                mse_scores.append(mse.cpu().numpy())
                mae_scores.append(mae.cpu().numpy())

                gt_img_inverse = gt_img_inverse.cpu().numpy()
                sr_img_inverse = sr_img_inverse.cpu().numpy()

                psnr_score = calculate_psnr(sr_img_inverse, gt_img_inverse)
                ssim_score = calculate_ssim(sr_img_inverse, gt_img_inverse)

                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)

                # plot img
                if is_plt:
                    time_index = indexs[index_count]
                    valid_time = get_sample_time(time_index)
                    img_name = valid_time + '.png'
                    img_outdir = os.path.join(img_outpath,img_name)

                    plt_discrete_indexs_img(lr_img_inverse.cpu().numpy(),gt_img_inverse,sr_img_inverse,img_outdir)
                else:
                    pass

                # valid

                # sr_img = tensor2np(sr[j])
                # gt_img = tensor2np(hr[j])

                # class_labels_img = class_labels[j].cpu().numpy().astype(np.int64)
                # pred_labels = statistics_class(sr_img_inverse)



                # acc, c_m, p_s, r_s, f1 = cal_class_score(class_labels_img, pred_labels)
                # csi = calculate_csi(sr_img_inverse, gt_img_inverse)

                # csi_scores.append(csi)
                # c_m_scores.append(c_m)

                # acc_scores.append(acc)
                # p_s_scores.append(p_s)
                # r_s_scores.append(r_s)
                # f1_scores.append(f1)

                index_count += 1
            tq_valid.set_postfix({'rmse': np.mean(rmse_scores), 'mae': np.mean(mae_scores)})
    tq_valid.close()
    # writer.add_image('val_img', sr_img_inverse, epoch)
    mean_psnr_scores = np.mean(psnr_scores)
    mean_ssim_scores = np.mean(ssim_scores)
    # mean_acc_scores = np.mean(acc_scores)
    # # print(f'p_s_scores.shape: {p_s_scores.shape}')
    # mean_p_s_scores = np.mean(p_s_scores, axis=0)
    # # print(f'r_s_scores.shape: {r_s_scores.shape}')
    # mean_r_s_scores = np.mean(r_s_scores, axis=0)
    # # print(f'f1_scores.shape: {f1_scores.shape}')
    # mean_f1_scores = np.mean(f1_scores, axis=0)
    # # print(f'csi_scores.shape: {csi_scores.shape}')
    # mean_csi_scores = np.mean(csi_scores, axis=0)
    mean_rmse_score = np.mean(rmse_scores)
    mean_mae_score = np.mean(mae_scores)
    writer.add_scalar('val_rmse', mean_rmse_score, epoch)
    writer.add_scalar('val_mae', mean_mae_score, epoch)

    log_info = f'[Validation] epoch: {epoch}, PSNR: {mean_psnr_scores:.4f}, ' \
               f'SSIM: {mean_ssim_scores:.4f}'\
               f'RMSE:{mean_rmse_score},MAE:{mean_mae_score}'
    '''
    ,Accuracy: {mean_acc_scores:.4f},' \
               f'Precision: {mean_p_s_scores},Recall: {mean_r_s_scores},' \
               f'F1: {mean_f1_scores},CSI: {mean_csi_scores},' \
    '''
    print(log_info)
    logger.info(log_info)

    return mean_ssim_scores


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                      test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
