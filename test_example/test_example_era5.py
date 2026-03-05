import sys
import os
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
import yaml
import json
import numpy as np
import torch.nn.utils as nn_utils
from sklearn.metrics import confusion_matrix
from util.utils import *
from models.all_elements.LU_SSD import IntensityClassifier, IntensityUpsampler


class Test_example_era5():
    def __init__(self,
                 par,
                 config,
                 model,
                 test_loader,
                 logger,
                 element_name
                 ):

        self.par = par
        self.config = config
        self.model = model

        self.logger = logger

        self.data_norm_mode = self.par.data_norm_mode
        self.is_class = self.config.model_config['cls']

        self.test_loader = test_loader

        self.best_ck_dir = self.par.best_model_path
        self.test_img_outpath = self.par.save_test_img_path

        self.lr_statistics_path = self.par.lr_statistics_path
        self.hr_statistics_path = self.par.hr_statistics_path

        self.result_scores = os.path.join(self.par.save_results_path, 'result.json')
        self.save_test_image_data_path = self.par.save_img_data

        self.hr_tp_mean, self.hr_tp_std = get_mean_std(self.hr_statistics_path)
        self.lr_tp_mean, self.lr_tp_std = get_mean_std(self.lr_statistics_path)

        self.element_name = element_name

    def test(self):
        torch.set_grad_enabled(False)

        self.load_model_state_dict(self.best_ck_dir)
        self.model.eval()

        index_count = 0

        psnr_scores = []
        ssim_scores = []

        rmse_scores = []
        mae_scores = []
        mse_scores = []


        dic_score = {}

        tq_test = tqdm(self.test_loader, desc=f'Test ', mininterval=0.3)

        tmp_sr = []
        tmp_lr_tp = []
        tmp_gt = []

        for lr, hr, geo_lsm, _,_ in tq_test:
            with torch.no_grad():
                if self.element_name=='tp':
                    lr = lr[:, 0, :, :].unsqueeze(1).to(torch.device('cuda'))
                elif self.element_name=='all':
                    lr = lr.to(torch.device('cuda'))
                else:
                    ValueError('element_name must be tp or all')

                hr = hr[:, 0, :, :].unsqueeze(1).to(torch.device('cuda'))
                geo_lsm = geo_lsm.to(torch.device('cuda'))
                if self.is_class:
                    sr, sr_label = self.model(lr, geo_lsm)
                else:
                    sr = self.model(lr, geo_lsm)

                lr_img = lr[0, 0, :, :]
                sr_img = sr[0, 0, :, :]
                gt_img = hr[0, 0, :, :]
                # Inverse Normlization

                lr_img_inverse = Inverse_Normlize_Precipitation_2(lr_img, self.lr_tp_mean, self.lr_tp_std)
                gt_img_inverse = Inverse_Normlize_Precipitation_2(gt_img, self.hr_tp_mean, self.hr_tp_std)

                lr_img_inverse = correct_data(lr_img_inverse)
                sr_img_inverse = correct_data(sr_img)
                gt_img_inverse = correct_data(gt_img_inverse)


                tmp_lr_tp.append(lr_img_inverse.cpu().numpy())
                tmp_sr.append(sr_img_inverse.cpu().numpy())
                tmp_gt.append(gt_img_inverse.cpu().numpy())
                img_dir =os.path.join(self.test_img_outpath,f'{str(index_count)}.png')
                plt_discrete_indexs_img(lr_img_inverse.cpu().numpy(),gt_img_inverse.cpu().numpy(),
                                            sr_img_inverse.cpu().numpy(),img_dir)

                rmse, mse = calculate_rmse_mse(sr_img_inverse, gt_img_inverse)
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
                index_count += 1
                tq_test.set_postfix({'rmse': np.mean(rmse_scores), 'mae': np.mean(mae_scores)})

        tq_test.close()

        sr_data = np.array(tmp_sr)
        gt_data = np.array(tmp_gt)
        lr_tp_data = np.array(tmp_lr_tp)

        sr_data_outdir = os.path.join(self.save_test_image_data_path, 'sr_data.npy')
        gt_data_outdir = os.path.join(self.save_test_image_data_path, 'gt_data.npy')
        lr_tp_data_outdir = os.path.join(self.save_test_image_data_path, 'lr_tp_data.npy')

        np.save(sr_data_outdir, sr_data)
        np.save(gt_data_outdir, gt_data)
        np.save(lr_tp_data_outdir, lr_tp_data)

        # writer.add_image('val_img', sr_img_inverse, epoch)
        mean_psnr_scores = np.mean(psnr_scores)
        mean_ssim_scores = np.mean(ssim_scores)

        mean_rmse_score = np.mean(rmse_scores)
        mean_mae_score = np.mean(mae_scores)
        mean_mse_score = np.mean(mse_scores)

        dic_score['psnr'] = float(mean_psnr_scores)
        dic_score['ssim'] = float(mean_ssim_scores)
        dic_score['rmse'] = float(mean_rmse_score)
        dic_score['mae'] = float(mean_mae_score)
        dic_score['mse'] = float(mean_mse_score)

        log_info = f'[Test] PSNR: {mean_psnr_scores:.4f}, ' \
                   f'SSIM: {mean_ssim_scores:.4f}, MSE:{mean_mse_score}, ' \
                   f'RMSE:{mean_rmse_score},MAE:{mean_mae_score},Max_Allocated_Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3)} GB'
        save_results(dic_score, self.result_scores)
        print(log_info)
        self.logger.info(log_info)

        cal_params_flops_test(self.model, (lr, geo_lsm), self.logger)

    def load_model_state_dict(self, path):
        if not os.path.exists(path):
            ValueError("Not find {}".format(path))
        ck = torch.load(path, map_location=torch.device('cuda'))

        self.model.load_state_dict(ck)

        # self.generator.load_state_dict({k.replace('module.', ''): v for k, v in ck.items()})
        print("Successfully load the trained model from {}".format(path))


