import os
import sys
import copy
import torch
import argparse
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.cuda.amp import GradScaler
from models.all_elements.LU_SSD import  LU_SSD

# from engines.engine import *
# from engines.engine_1 import *
from test_example.test_example_era5 import Test_example_era5
from util.utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"
from util.utils import *
# from configs.config_setting_110 import *
from configs.config_setting_111 import *
# from configs.config_setting_100 import *
# from configs.config_setting_101 import *
# from configs.config_setting_011 import *
# from configs.config_setting_010 import *
from dataloader.dataloader import GetLoader
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

def main(config,par):
    print("~~~~~~~~~~~~~~~~current scale is: X{} ~~~~~~~~~~~~~~~~~~~~~".format(config.scale))
    print('#----------config----------#')
    print(config)
    print(par)
    print('#----------Creating logger----------#')

    global logger
    logger = get_logger('train', par.log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    n_gpu = torch.cuda.device_count()
    print(f"number of gpu is: {n_gpu}")
    set_seed(config.seed)
    torch.cuda.empty_cache()

    test_data_path = os.path.join(config.test_data_path, f'test_example_era5.hdf5')

    print('#----------Preparing dataset----------#')

    test_loader = GetLoader(test_data_path,
                                batch_size=1,
                                num_workers=par.num_workers,
                                shuffle_is_true = False)


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config

    c_list = [64, 128, 256, 512]

    logger.info(f'c_list: {c_list}')

    if par.element_name == 'tp':
        input_channels=1
        from models.only_tp.LU_SSD import LU_SSD
    elif par.element_name == 'all':
        from models.all_elements.LU_SSD import LU_SSD
        input_channels=21
    else:
        ValueError('element_name must be tp or all')

    model = LU_SSD(input_channels=input_channels,
                    out_channels=model_cfg['num_classes'],
                    rs_factor = config.scale,
                    c_list=c_list,
                    res = model_cfg['residual'],
                    is_cls = model_cfg['cls'],
                    split_nums = model_cfg['splict_nums'],
                    atten_config=model_cfg['dic_atten'],
                    ssd_config = model_cfg['dic_ssd'],
                    dic_geo_lsm = model_cfg['dic_geo_lsm'],
                    act_name='relu').to(torch.device('cuda'))

    Model_test = Test_example_era5(par,
                          config,
                          model,
                          test_loader,
                          logger,
                          par.element_name
                          )
    return Model_test


if __name__ == '__main__':

    dataset_name = 'test_example'
    parser = argparse.ArgumentParser(description='LU-SSD')

    parser.add_argument('--mode', type=str, default='111',
                        help='mode of the model')
    parser.add_argument('--split_num', type=int, default=8,
                        help='number of split channel')
    # Hardware specifications
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads for data loading')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--element_name', type=str, default='all',
                        help='tp or all')
    # train setting
    parser.add_argument('--data_norm_mode', type=str, default='norm',
                        help='data process mode')

    parser.add_argument('--lr_statistics_path', type=str, default=f'./data/era5_{dataset_name}.json',
                        help='lr dataset mean, std file')
    parser.add_argument('--hr_statistics_path', type=str, default=f'./data/{dataset_name}.json',
                        help='hr dataset mean, std file')

    parser.add_argument('--log_dir', type=str, default='',
                        help='train log path for train')

    parser.add_argument('--test_root_path', type=str,
                        default='/test',
                        help='root path for test,must write this')
    parser.add_argument('--save_test_img_path', type=str, default='',
                        help='save sr image')
    parser.add_argument('--save_img_data', type=str, default='',
                        help='save sr data')
    parser.add_argument('--save_gt_path', type=str, default='',
                        help='save gt data')
    parser.add_argument('--save_results_path', type=str, default='',
                        help='save test results')
    parser.add_argument('--best_model_path', type=str, default='', help=' path of best model')


    par = parser.parse_args()


    if par.mode == '110':
        if par.split_num == 1:
            config = setting_110_1ele_in1_out1_split1_x8
        elif par.split_num == 2:
            config = setting_110_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_110_3ele_in3_out1_split4_x8
        elif par.split_num == 8:
            config = setting_110_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '000':
        if par.split_num == 1:
            # config = setting_1ele_in1_out1_split1_x8_001
            config = setting_000_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '001':
        if par.split_num == 1:
            # config = setting_1ele_in1_out1_split1_x8_001
            config = setting_001_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '010':
        if par.split_num == 1:
            config = setting_010_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '100':
        if par.split_num == 1:
            # config = setting_1ele_in1_out1_split1_x8_001
            config = setting_100_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '011':
        if par.split_num == 1:
            # config = setting_1ele_in1_out1_split1_x8_001
            config = setting_011_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '101':
        if par.split_num == 1:
            # config = setting_1ele_in1_out1_split1_x8_001
            config = setting_101_21ele_in19_out1_split1_x8
        elif par.split_num == 2:
            config = setting_1ele_in1_out1_split2_x8
        elif par.split_num == 4:
            config = setting_1ele_in1_out1_split4_x8
        elif par.split_num == 8:
            config = setting_1ele_in1_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    elif par.mode == '111':
        if par.split_num == 1:
            config = setting_111_21ele_in21_out1_split1_x8
        elif par.split_num == 2:
            config = setting_111_21ele_in21_out1_split2_x8
        elif par.split_num == 4:
            config = setting_111_21ele_in21_out1_split4_x8
        elif par.split_num == 8:
            config = setting_111_21ele_in21_out1_split8_x8
        elif par.split_num == 16:
            config = setting_1ele_in1_out1_split8_x8

        else:
            print("Error split_num, must be 1,2,4,8 or 16 !!!")
    else:
        print("Error Mode, must be '000', '001', '101', '100', '011', '110', '101', or '111' !!!")
    par.best_model_path = os.path.join(par.test_root_path, 'checkpoint', 'checkpoint_best.pth')
    par.test_root_path = os.path.join(par.test_root_path,'test_example/era5',par.element_name)
    par.save_img_data = os.path.join(par.test_root_path, 'img_data')
    par.save_test_img_path = os.path.join(par.test_root_path, 'img')

    par.save_results_path = os.path.join(par.test_root_path, 'results')
    par.log_dir = os.path.join(par.test_root_path, 'log')

    os.makedirs(par.log_dir, exist_ok=True)
    os.makedirs(par.save_img_data, exist_ok=True)
    os.makedirs(par.save_test_img_path, exist_ok=True)
    os.makedirs(par.save_results_path, exist_ok=True)

    config.num_workers = par.num_workers


    Test_example = main(config,par)

    Test_example.test()
