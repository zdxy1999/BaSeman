import csv
import os
import random
import time

import pandas as pd
import numpy as np
import torch.utils.data
import time
import netCDF4
import torchvision.transforms
from skimage import io
from PIL import Image
import cv2

import utils.augmentation
from utils.aug_plus import augmentation_plus_torch
from utils.aug_plus import augmentation_plus
from dataloaders.data_utils import get_unk_mask_indices, image_loader


def read_rs_to_numpy(in_file):
    with netCDF4.Dataset(in_file) as nf:
        rs_01 = nf.variables["albedo_01"][:].data
        rs_02 = nf.variables["albedo_02"][:].data
        rs_03 = nf.variables["albedo_03"][:].data
        rs_04 = nf.variables["albedo_04"][:].data
        rs_05 = nf.variables["albedo_05"][:].data
        rs_06 = nf.variables["albedo_06"][:].data
        rs_07 = nf.variables["tbb_07"][:].data
        rs_08 = nf.variables["tbb_08"][:].data
        rs_09 = nf.variables["tbb_09"][:].data
        rs_10 = nf.variables["tbb_10"][:].data
        rs_11 = nf.variables["tbb_11"][:].data
        rs_12 = nf.variables["tbb_12"][:].data
        rs_13 = nf.variables["tbb_13"][:].data
        rs_14 = nf.variables["tbb_14"][:].data
        rs_15 = nf.variables["tbb_15"][:].data
        rs_16 = nf.variables["tbb_16"][:].data

    hsi = np.array((rs_01, rs_02, rs_03, rs_04, rs_05, rs_06, rs_07, rs_08,
                    rs_09, rs_10, rs_11, rs_12, rs_13, rs_14, rs_15, rs_16))
    return hsi

def geo_transform(year, month, day, loc_y, loc_x, L, W, step, patch_l, patch_w, range_l, range_w, start_lat, start_lon):
    '''
        transform coordinate into longitude and latitude, time into year time
        the output data are dimansionless
        :param year:
        :param month:
        :param day:
        :param loc_y: coordinate y in image name
        :param loc_x: coordinate x in image name
        :param L: pixel length of whole disc data
        :param W: pixel width of whole disc data
        :param step: pixel step of patch
        :param patch_l: length of patch
        :param patch_w: width of patch
        :param range_l: longitude range of whole disc data
        :param range_w: latitude range of whole disc data
        :param start_lat: the latitude of up-left point of whole disc data
        :param start_lon: the longitude of up-left point of whole disc data
        :return: tuple (year_time, y_result, x_result, l_result, w_result), elements are torch.tensors
    '''

    month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_prior_normal = [0, 31, 59, 90, 120, 151, 182, 213, 244, 274, 305, 335]
    month_prior_leap = [0, 31, 59 + 1, 90 + 1, 120 + 1, 151 + 1, 182 + 1, 213 + 1, 244 + 1, 274 + 1, 305 + 1, 335 + 1]

    leap_mark = year % 4 == 0

    prior_day = 0
    year_day = 0
    year_len = 0
    # print(year_day.shape)
    if leap_mark:
        prior_day = month_prior_leap[month - 1]
        year_len = 366
    else:
        prior_day = month_prior_normal[month - 1]
        year_len = 365
    year_day = prior_day + day
    year_time = year_day / year_len
    year_time = torch.Tensor([year_time])

    y = start_lat - ((loc_y * step + 0.5 * patch_l) / L * range_l)
    x = start_lon + ((loc_x * step + 0.5 * patch_w) / W * range_w)

    if x > 180:
        x = x - 360

    y_result = torch.tensor([y - (-90) / 180], dtype=torch.float32)
    x_result = torch.tensor([(x - (-180)) / 360], dtype=torch.float32)
    l_result = torch.tensor([(patch_l / L * range_l) / 180], dtype=torch.float32)
    w_result = torch.tensor([(patch_w / W * range_w) / 360], dtype=torch.float32)

    output = torch.concat((year_time, y_result, x_result, l_result, w_result))

    return output  # (year_time, y_result, x_result, l_result, w_result)


class MLC_Dataset_2401(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, num_labels ,known_labels=0,transform=None,testing=False,tk_ratio=0.25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        # ground truth
        self.labels_frame = pd.read_csv(csv_file)

        # for ctran training
        self.tk_ratio = tk_ratio

        # img dir
        self.root_dir = root_dir

        # transform
        self.transform = transform
        self.testing = testing
        self.num_labels = num_labels
        self.known_labels = known_labels

        # for multiworkers
        self.start = 0
        self.end = len(self.labels_frame) # no need to -1

        # file_path_list
        self.file_path = self.root_dir + '/NC_H08_20190101_0020_R21_FLDK.02401_02401.nc'

        self.whole_hsi = read_rs_to_numpy(self.file_path)
        self.ymd_str = '20190101'
        self.max = self.whole_hsi.reshape(16,-1).max(axis=1).reshape(16,1,1)
        self.min = self.whole_hsi.reshape(16,-1).min(axis=1).reshape(16,1,1)
        self.whole_hsi = (self.whole_hsi-self.min)/(self.max-self.min)


    def __len__(self):
        # get the length of data
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_name = self.labels_frame.iloc[idx, 0].replace('.png', '')
        patch_ymd = patch_name[0:8]
        loc_x,loc_y = int(patch_name[-2:])-1,int(patch_name[-5:-3])-1
        img_month = int(patch_name[4:6])-1
        #print(img_month)

        # update data
        if patch_ymd != self.ymd_str:
            self.ymd_str = patch_ymd
            self.file_path = self.root_dir + '/NC_H08_%s_0020_R21_FLDK.02401_02401.nc' % patch_ymd
            #self.whole_hsi = np.load(self.file_path)
            self.whole_hsi = read_rs_to_numpy(self.file_path)
            self.max = self.whole_hsi.reshape(16, -1).max(axis=1).reshape(16, 1, 1)
            self.min = self.whole_hsi.reshape(16, -1).min(axis=1).reshape(16, 1, 1)
            self.whole_hsi=(self.whole_hsi-self.min)/(self.max-self.min)#*2-1
            arr = self.whole_hsi.reshape((16,-1))
            #print(np.min(arr,axis=1))

        step = 80
        w = 400
        patch = self.whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]

        patch = torch.from_numpy(patch.copy())
        # patch = torchvision.transforms.Resize(256)(patch)

        if not self.testing:
            patch = utils.aug_plus.augmentation_plus_torch(patch)
        patch = patch * 2 - 1

        img_loc = [loc_x, loc_y]

        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        image_id = self.labels_frame.iloc[idx, 0]
        sample = {'image': patch, 'labels': labels}

        if self.transform:
            hsi = self.transform(patch)
            labels = torch.Tensor(labels)

        # for ctran
        mask = labels.clone()
        unk_mask_indices = get_unk_mask_indices(hsi, self.testing, self.num_labels, 0, 0)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        caption = np.array(range(17))
        caption = caption * labels.numpy()
        caption[caption == 0] = 17
        caption.sort()
        caption = torch.LongTensor(caption)

        sample['image'] = hsi
        sample['labels'] = labels
        sample['mask'] = mask
        sample['image_loc'] = img_loc

        sample['loc_num'] = (img_loc[0]) * 11 + (img_loc[1])
        sample['month'] = img_month
        return sample

class MLC_Dataset_2401_shuffle_aug(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, num_labels ,known_labels=0,transform=None,testing=False,tk_ratio=0.25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        # ground truth
        self.labels_frame = pd.read_csv(csv_file)

        # for ctran training
        self.tk_ratio = tk_ratio

        # img dir
        self.root_dir = root_dir

        # transform
        self.transform = transform
        self.testing = testing
        self.num_labels = num_labels
        self.known_labels = known_labels

        # for multiworkers
        self.start = 0
        self.end = len(self.labels_frame) # no need to -1

        # file_path_list
        random_flag = np.random.randint(low=1, high=4, size=1)
        self.file_path = self.root_dir + '/2019_00%01d0/NC_H08_20190101_00%1d0_R21_FLDK.02401_02401.nc' % (random_flag,random_flag)
        #self.file_path = self.root_dir + '/NC_H08_20190101_0020_R21_FLDK.02401_02401.nc'

        self.whole_hsi = read_rs_to_numpy(self.file_path)
        self.ymd_str = '20190101'
        self.max = self.whole_hsi.reshape(16,-1).max(axis=1).reshape(16,1,1)
        self.min = self.whole_hsi.reshape(16,-1).min(axis=1).reshape(16,1,1)
        self.whole_hsi = (self.whole_hsi-self.min)/(self.max-self.min)


    def __len__(self):
        # get the length of data
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_name = self.labels_frame.iloc[idx, 0].replace('.png', '')
        patch_ymd = patch_name[0:8]
        loc_x,loc_y = int(patch_name[-2:])-1,int(patch_name[-5:-3])-1
        img_month = int(patch_name[4:6])-1
        #print(img_month)

        # update data
        if patch_ymd != self.ymd_str:
            self.ymd_str = patch_ymd

            if not self.testing:
                random_flag = np.random.randint(low=1, high=4, size=1)
                self.file_path = self.root_dir + '/2019_0020/NC_H08_%s_0020_R21_FLDK.02401_02401.nc' % patch_ymd
                self.file_path_aux = self.root_dir + '/2019_00%01d0/NC_H08_%s_00%01d0_R21_FLDK.02401_02401.nc' % (
                random_flag, patch_ymd, random_flag)

                while not os.path.exists(self.file_path_aux):
                    random_flag = np.random.randint(low=1, high=4, size=1)
                    #self.file_path = self.root_dir + '/NC_H08_20190101_00%1d0_R21_FLDK.02401_02401.nc' % (random_flag)
                    self.file_path_aux = self.root_dir + '/2019_00%01d0/NC_H08_%s_00%01d0_R21_FLDK.02401_02401.nc' % (random_flag,patch_ymd,random_flag)
                    self.file_path = self.root_dir + '/2019_0020/NC_H08_%s_0020_R21_FLDK.02401_02401.nc' % (patch_ymd)
                    #self.whole_hsi = np.load(self.file_path)
            else:
                self.file_path = self.root_dir + '/2019_0020/NC_H08_%s_0020_R21_FLDK.02401_02401.nc' % (patch_ymd)

            if not self.testing:
                p = torch.rand(1)
                a = max(1-p,p)
                img_1 = read_rs_to_numpy(self.file_path)
                img_2 = read_rs_to_numpy(self.file_path_aux)
                self.whole_hsi = a*img_1 + (1-a)*img_2
            else:
                self.whole_hsi = read_rs_to_numpy(self.file_path)

            self.whole_hsi = read_rs_to_numpy(self.file_path)
            self.max = self.whole_hsi.reshape(16, -1).max(axis=1).reshape(16, 1, 1)
            self.min = self.whole_hsi.reshape(16, -1).min(axis=1).reshape(16, 1, 1)
            self.whole_hsi=(self.whole_hsi-self.min)/(self.max-self.min)#*2-1
            arr = self.whole_hsi.reshape((16,-1))

        step = 80
        w = 400
        patch = self.whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]

        patch = torch.from_numpy(patch.copy())
        patch = torchvision.transforms.Resize(256)(patch)

        if not self.testing:
            patch = utils.aug_plus.augmentation_plus_torch(patch)
        patch = patch*2-1

        img_loc = [loc_x, loc_y]

        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        image_id = self.labels_frame.iloc[idx, 0]
        sample = {'image': patch, 'labels': labels}

        if self.transform:
            hsi = self.transform(patch)
            labels = torch.Tensor(labels)

        # for ctran
        mask = labels.clone()
        unk_mask_indices = get_unk_mask_indices(hsi, self.testing, self.num_labels, 0, 0)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        caption = np.array(range(17))
        caption = caption * labels.numpy()
        caption[caption == 0] = 17
        caption.sort()
        caption = torch.LongTensor(caption)

        sample['image'] = hsi
        sample['labels'] = labels
        sample['mask'] = mask
        sample['image_loc'] = img_loc

        sample['loc_num'] = (img_loc[0]) * 11 + (img_loc[1])
        sample['month'] = img_month
        return sample








