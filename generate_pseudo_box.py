import argparse
import collections
import os
import os.path as osp
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
from dataset.prostate import Prostate_Multi, Prostate_Multi2
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

# from models.model_zoo import get_model

from tqdm import tqdm
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel, Upsample
import SimpleITK as sitk
import warnings

warnings.filterwarnings('ignore')


def bbox(img):
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return rmin, rmax, cmin, cmax


def maxAreaOfIsland(grid) -> int:
    ans = 0
    for i, l in enumerate(grid):
        for j, n in enumerate(l):
            cur = 0
            q = collections.deque([(i, j)])
            while q:
                cur_i, cur_j = q.popleft()
                if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                    continue
                cur += 1
                grid[cur_i][cur_j] = 0
                for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    next_i, next_j = cur_i + di, cur_j + dj
                    q.append((next_i, next_j))
            ans = max(ans, cur)
    return ans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None, required=True, help='Model path')
    parser.add_argument('--dataset', type=str, default='prostate', help='training dataset')
    parser.add_argument('--data_dir', default='/data/yhz', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--test_prediction_save_path', type=str, default='./results5',
                        help='Path root for test image and mask')
    parser.add_argument('--save_result', action='store_true', help='Save Results')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze Batch Normalization')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def main(args):
    print(args.model_file)

    if not os.path.exists(args.test_prediction_save_path):
        os.makedirs(args.test_prediction_save_path)

    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    encoder = Encoder(c=args.in_channels, norm=args.norm, activation=args.activation)
    seg_decoder = Decoder(num_classes=args.num_classes, norm=args.norm, activation=args.activation)

    state_dicts = torch.load(model_file)

    encoder.load_state_dict(state_dicts['encoder_state_dict'])
    seg_decoder.load_state_dict(state_dicts['seg_decoder_state_dict'])

    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()
    if not args.freeze_bn:
        encoder.eval()
        for m in encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        seg_decoder.eval()
        for m in seg_decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        encoder.eval()
        seg_decoder.eval()

    dataset_val = Prostate_Multi2(base_dir='/data/yhz/prostate', split='train', domain_idx_list=[args.datasetTest])
    dataset_val_loader = DataLoader(dataset_val, batch_size=1, num_workers=8,
                                    shuffle=False, drop_last=False, pin_memory=True)
    epoch = 0
    invalid = 0
    valid = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataset_val_loader):
            epoch += 1

            image = sample[0]
            label = sample[1].detach().data.cpu().numpy()
            path = sample[3][0]
            pred_student = torch.max(torch.softmax(seg_decoder(encoder(image)), dim=1), dim=1)[
                1].detach().data.cpu().numpy()


            rmin, rmax, cmin, cmax = bbox(label[0])
            a = cmin
            b = rmin
            c = cmax - cmin
            d = rmax - rmin
            print('gt', a, b, c, d)
            rmin, rmax, cmin, cmax = bbox(pred_student[0])
            a = cmin
            b = rmin
            c = cmax - cmin
            d = rmax - rmin
            print('pd', a, b, c, d)
            listpred = pred_student[0].tolist()
            maxland = maxAreaOfIsland(listpred)
            allland = np.sum(pred_student[0])
            trueland = np.sum(label[0])

            print(trueland, maxland, allland)
            if maxland/allland<0.9:
                print('invalid')
                invalid+=1
                continue
            valid+=1
            if args.save_result:
                path = path.replace('image', 'mask').replace('npy', 'txt').replace('prostate','prostate_pseudo_box')
                print(path)
                file_dir = os.path.split(path)[0]

                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                if not os.path.exists(path):
                    os.system(r'touch %s' % path)
                Note = open(path, mode='w')
                Note.write(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d))
                Note.close()

    print('valid',valid,'invalid',invalid)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
    main(args)
