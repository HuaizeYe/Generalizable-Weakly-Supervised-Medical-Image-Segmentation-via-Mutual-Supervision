#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn as nn
import torch
from utils.metrics import *
from dataset import utils
from utils.utils import save_per_img_prostate, _connectivity_region_analysis
from test_utils import *
from networks.unet import Encoder, Decoder
from tqdm import tqdm
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel
import SimpleITK as sitk
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Test on Prostate dataset (3D volume)')
    # basic settings
    parser.add_argument('--dataset', type=str, default='prostate', help='training dataset')
    parser.add_argument('--data_dir', default='/data/yhz', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def main(args):
    domain_name = domain_list[args.datasetTest]
    data_dir = os.path.join(args.data_dir, args.dataset)

    file_list = [item for item in os.listdir(os.path.join(data_dir, domain_name)) if 'segmentation' not in item]


    tbar = tqdm(file_list, ncols=150)

    for file_name in tbar:
        itk_image = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name))
        itk_mask = sitk.ReadImage(
            os.path.join(data_dir, domain_name, file_name.replace('.nii.gz', '_segmentation.nii.gz')))

        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)

        max_value = np.max(image)
        min_value = np.min(image)
        image = 2 * (image - min_value) / (max_value - min_value) - 1

        mask[mask == 2] = 1

        #### channel 3 ####
        frame_list = [kk for kk in range(1, image.shape[0] - 1)]
        count = 0
        for ii in range(int(np.floor(image.shape[0] // args.batch_size))):
            for idx, jj in enumerate(frame_list[ii * args.batch_size: (ii + 1) * args.batch_size]):
                img = image[jj - 1: jj + 2, ...].copy()
                mas = mask[jj, ...]
                img = img.transpose(1, 2, 0)
                count += 1
                print(file_name,count)
                # if count < 10:
                #     imagepath = '/data/yhz/prostate_test/Domain' + str(
                #         args.datasetTest + 1) + '/image/' + file_name.replace('.nii.gz', '_' +
                #                                                               '0' + str(count) + '.npy')
                # else:
                #     imagepath = '/data/yhz/prostate_test/Domain' + str(
                #         args.datasetTest + 1) + '/image/' + file_name.replace('.nii.gz', '_' + str(count) + '.npy')
                # maskpath = imagepath.replace('image', 'mask')
                # file_dir = os.path.split(imagepath)[0]
                # if not os.path.isdir(file_dir):
                #     os.makedirs(file_dir)
                # file_dir = os.path.split(maskpath)[0]
                # if not os.path.isdir(file_dir):
                #     os.makedirs(file_dir)
                # np.save(imagepath, img)
                # np.save(maskpath, mas)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
    main(args)
