import os
import torch
import random
import h5py
from util.utils import get_sample_time
from torch.utils import data
from torchvision import transforms as T

import numpy as np
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.length = len(self.loader)
        self.stream = torch.cuda.Stream()

        # # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.lr, self.hr,self.geo_lsm,self.class_labels = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.lr, self.hr, self.geo_lsm,self.class_labels = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.hr = self.hr.cuda(non_blocking=True)
            self.lr = self.lr.cuda(non_blocking=True)
            self.geo_lsm = self.geo_lsm.cuda(non_blocking=True)
            self.class_labels = self.class_labels.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.__preload__()
        return self.lr, self.hr, self.geo_lsm,self.class_labels

    def __len__(self):
        """Return the number of images."""
        return self.length


class HDF5Dataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                 h5_path,
                 data_transform=None,
                 seed=1234,
                 shuffle = False):
        """Initialize and preprocess the lmdb dataset."""
        self.h5_path = h5_path
        self.h5file = h5py.File(h5_path, 'r')
        if not self.h5file.__contains__("__len__"):
           print("Error")
        self.keys = self.h5file["__len__"][()]  # 86366
        # self.keys   = self.keys[0]
        self.length = self.keys
        self.data_transform = data_transform
        self.keys = [str(k) for k in range(self.keys)]
        random.seed(seed)
        if shuffle:
            random.shuffle(self.keys)

    def __getitem__(self, index):
        """Return low-resolution frames and its corresponding high-resolution."""
        iii = self.keys[index]
        hr = self.h5file["hr_"+iii][()]
        # lr = self.h5file[iii + "lr"][()]
        lr = self.h5file["lr_"+iii][()]
        class_labels = self.h5file["classlabel_"+iii][()]
        lr_class_labels = self.h5file["lr_classlabel_" + iii][()]
        geo_lsm = self.h5file["geo"][()]
        if self.data_transform is not None:
            hr = self.data_transform(hr)
            lr = self.data_transform(lr)
            geo_lsm = self.data_transform(geo_lsm)

        return lr, hr, geo_lsm,class_labels,lr_class_labels

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.h5_path + ')'

def GetLoader(hdf5_dir,
              batch_size=64,
              random_seed=1234,
              num_workers=8,
              shuffle_is_true=False):
    """Build and return a data loader."""

    c_transforms = []

    c_transforms.append(T.RandomHorizontalFlip())

    c_transforms.append(T.RandomVerticalFlip())

    c_transforms = T.Compose(c_transforms)

    c_transforms = None

    content_dataset = HDF5Dataset(hdf5_dir, c_transforms, random_seed, shuffle=shuffle_is_true)
    content_data_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size,
                                          drop_last=True, shuffle=shuffle_is_true, num_workers=num_workers, pin_memory=True)
    # content_data_loader = DataPrefetcher(content_data_loader)
    return content_data_loader
class HDF5Dataset_IFS(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                 h5_path,
                 data_transform=None,
                 seed=1234,
                 shuffle = False):
        """Initialize and preprocess the lmdb dataset."""
        self.h5_path = h5_path
        self.h5file = h5py.File(h5_path, 'r')
        if not self.h5file.__contains__("__len__"):
           print("Error")
        self.keys = self.h5file["__len__"][()]  # 86366
        # self.keys   = self.keys[0]
        self.length = self.keys
        self.data_transform = data_transform
        self.keys = [str(k) for k in range(self.keys)]
        random.seed(seed)
        if shuffle:
            random.shuffle(self.keys)

    def __getitem__(self, index):
        """Return low-resolution frames and its corresponding high-resolution."""
        iii = self.keys[index]

        lr = self.h5file["lr_"+iii][()]

        if self.data_transform is not None:
            lr = self.data_transform(lr)

        return lr

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.h5_path + ')'
def GetLoader_ifs(hdf5_dir,
              batch_size=16,
              random_seed=1234,
              num_workers=8,
              shuffle_is_true=False):
    """Build and return a data loader."""

    c_transforms = []

    c_transforms.append(T.RandomHorizontalFlip())

    c_transforms.append(T.RandomVerticalFlip())

    c_transforms = T.Compose(c_transforms)

    c_transforms = None

    content_dataset = HDF5Dataset_IFS(hdf5_dir, c_transforms, random_seed, shuffle=shuffle_is_true)
    content_data_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size,
                                          drop_last=True, shuffle=shuffle_is_true, num_workers=num_workers, pin_memory=True)
    # content_data_loader = DataPrefetcher(content_data_loader)
    return content_data_loader
if __name__ == '__main__':

    content_dataset = "/public/001/suqingguo/w2_weather/data_new/hdf5/valid_X8_3.hdf5"
    indexs_path = "/public/001/suqingguo/w2_weather/results" \
                       "/statistics/index_zarr_no_NAN_INF/valid_index_3.npy"
    data_loader = GetLoader(content_dataset,num_workers=4)
    out_path = '/public/001/suqingguo/w2_weather/results/statistics/evaluate/'
    count = 0

    indexs = np.load(indexs_path)

    for _ in range(len(data_loader)-1):
        lr, hr, geo_lsm,class_labels = data_loader.next()
        lr = lr[:, 0, :, :].cpu().numpy()
        hr = hr[:, 0, :, :].cpu().numpy()

        for i in range(hr.shape[0]):
            lr_img = lr[i]
            hr_img = hr[i]

            lr_img_inverse = Inverse_normlize_Precipitation(lr_img, 'lr')
            hr_img_inverse = Inverse_normlize_Precipitation(hr_img, 'gt')

            time_index = indexs[count]
            valid_time = get_sample_time(time_index)
            img_name = valid_time + '.png'
            img_outdir = os.path.join(out_path, img_name)

            # plt_img(lr_img_inverse, hr_img_inverse, img_outdir)
            #
            # count+=1
