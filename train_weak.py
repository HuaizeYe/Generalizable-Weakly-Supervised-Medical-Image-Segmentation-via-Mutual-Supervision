import os
import argparse
import sys

import numpy as np

from model.model import PrivateDecoder
from network import deeplabv3_resnet50
from networks.unet import Encoder, Decoder, Rec_Decoder
from utils.utils import count_params
from tensorboardX import SummaryWriter
import random
import dataset.transform as trans
from torchvision.transforms import Compose

from dataset.fundus import Fundus_Multi, Fundus, Fundus_Multi3
from dataset.prostate import Prostate_Multi, Prostate_Multi4
import torch.backends.cudnn as cudnn

from torch.nn import BCELoss, CrossEntropyLoss, DataParallel, KLDivLoss, MSELoss
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from utils.losses import dice_loss, dice_loss_multi
from utils.utils import decode_seg_map_sequence
import shutil
from utils.utils import postprocessing, _connectivity_region_analysis
from utils.metrics import *
import os.path as osp
import SimpleITK as sitk
from medpy.metric import binary
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

fundus_batch_list = [[3, 6, 7],
                     [2, 7, 7],
                     [2, 4, 10],
                     [2, 4, 10]]

prostate_batch_list = [[2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2]]

def create_bitmask(mask, bbox):
    per_im_bitmasks_full = []
    w, h = mask.shape[2], mask.shape[3]
    for i in range(mask.shape[0]):
        bitmask_full = torch.zeros((2, mask.shape[2], mask.shape[3])).cuda().float()
        for j in range(mask.shape[1]):
            bitmask_full[j][
            int((bbox[j]["ch"][i] - bbox[j]["h"][i] / 2) * h):int((bbox[j]["ch"][i] + bbox[j]["h"][i] / 2) * h + 1),
            int((bbox[j]["cw"][i] - bbox[j]["w"][i] / 2) * w):int(
                (bbox[j]["cw"][i] + bbox[j]["w"][i] / 2) * w + 1)] = 1.0
        per_im_bitmasks_full.append(bitmask_full)

    gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
    return gt_bitmasks_full


def compute_project_term(mask_scores, gt_bitmasks):
    def dice_coefficient(x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    ms = torch.empty(0).cuda()
    gt = torch.empty(0).cuda()
    for i in range(mask_scores.shape[1]):
        ms = torch.cat(([ms, mask_scores[:, i, :, :]]))
        gt = torch.cat(([gt, gt_bitmasks[:, i, :, :]]))
    ms = ms.unsqueeze(1)
    gt = gt.unsqueeze(1)
    mask_losses_y = dice_coefficient(
        ms.max(dim=2, keepdim=True)[0],
        gt.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        ms.max(dim=3, keepdim=True)[0],
        gt.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()

def parse_args():
    parser = argparse.ArgumentParser(description='DG Medical Segmentation Train')
    # basic settings
    parser.add_argument('--data_root', type=str, default='/data/yhz', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'], help='training dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default=3, help='training epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--lambda_rec', type=float,  default=0.1, help='lambda of rec')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--ram', action='store_true', help='whether use ram augmentation')
    parser.add_argument('--rec', action='store_true', help='whether use rec loss')
    parser.add_argument('--is_out_domain', action='store_true', help='whether use out domain amp')
    parser.add_argument('--consistency', action='store_true', help='whether use consistency loss')
    parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='path of saved checkpoints')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def KD(input, target):
    consistency_criterion = KLDivLoss()
    loss_consistency = consistency_criterion(input.log(), target) + consistency_criterion(target.log(), input)
    return loss_consistency


def test_fundus(encoder, epoch, data_dir, datasetTest, output_path, batch_size=8, dataset='fundus'):
    encoder.eval()
    data_dir = os.path.join(data_dir, dataset)
    transform = Compose([trans.Resize((256, 256)), trans.Normalize()])
    testset = Fundus(base_dir=data_dir, split='test',
                     domain_idx=datasetTest, transform=transform)
    
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)
    
    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_num = 0
    tbar = tqdm(testloader, ncols=150)

    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()

            prediction = torch.sigmoid(encoder(data)[1])
            prediction = torch.nn.functional.interpolate(prediction, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")

            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=dataset, threshold=0.75)
                cup_dice, disc_dice = dice_coeff_2label(prediction_post, target_orgin[i])
                val_cup_dice += cup_dice
                val_disc_dice += disc_dice
                total_num += 1
        val_cup_dice /= total_num
        val_disc_dice /= total_num

        print('val_cup_dice : {}, val_disc_dice : {}'.format(val_cup_dice, val_disc_dice))
        with open(osp.join(output_path, str(datasetTest) + '_val_log.csv'), 'a') as f:
            log = [['batch-size: '] + [batch_size] + [epoch] + \
                   ['cup dice coefficence: '] + [val_cup_dice] + \
                   ['disc dice coefficence: '] + [val_disc_dice]]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        
        return (val_cup_dice + val_disc_dice) * 100.0 / 2

def test_prostate(encoder,  epoch, data_dir, datasetTest, output_path, batch_size=8, dataset='prostate'):
    encoder.eval()
    domain_name = domain_list[datasetTest]
    data_dir = os.path.join(data_dir, dataset)

    file_list = [item for item in os.listdir(os.path.join(data_dir, domain_name)) if 'segmentation' not in item]

    tbar = tqdm(file_list, ncols=150)

    val_dice = 0.0
    total_num = 0
    for file_name in tbar:
        itk_image = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name))
        itk_mask = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name.replace('.nii.gz', '_segmentation.nii.gz')))

        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)

        max_value = np.max(image)
        min_value = np.min(image)
        image = 2 * (image - min_value) / (max_value - min_value) - 1

        mask[mask==2] = 1
        pred_y = np.zeros(mask.shape)

        #### channel 3 ####
        frame_list = [kk for kk in range(1, image.shape[0] - 1)]

        for ii in range(int(np.floor(image.shape[0] // batch_size))):
            vol = np.zeros([batch_size, 3, image.shape[1], image.shape[2]])

            for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                vol[idx, ...] = image[jj - 1 : jj + 2, ...].copy()
            vol = torch.from_numpy(vol).float().cuda()

            pred_student = torch.max(torch.softmax(encoder(vol)[1], dim=1), dim=1)[1].detach().data.cpu().numpy()

            for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                ###### Ignore slices without prostate region ######
                if np.sum(mask[jj, ...]) == 0:
                    continue
                pred_y[jj, ...] = pred_student[idx, ...].copy()

        
        processed_pred_y = _connectivity_region_analysis(pred_y)
        dice_coeff = binary.dc(np.asarray(processed_pred_y, dtype=bool),
                            np.asarray(mask, dtype=bool))
        val_dice += dice_coeff
        total_num += 1
    
    val_dice /= total_num
    print('val_dice : {}'.format(val_dice))
    with open(osp.join(output_path, str(datasetTest) + '_val_log.csv'), 'a') as f:
            log = [['batch-size: '] + [batch_size] + [epoch] + \
                   ['dice coefficence: '] + [val_dice]]
            log = map(str, log)
            f.write(','.join(log) + '\n')
    return val_dice * 100.0


def train_fundus(trainloader_list, encoder,  rec_decoder, writer, args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list):
    if args.consistency_type == 'mse':
        consistency_criterion = MSELoss()
    elif args.consistency_type == 'kl':
        consistency_criterion = KLDivLoss()
    elif args.consistency_type == 'kd':
        consistency_criterion = KD
    else:
        assert False, args.consistency_type
    criterion = BCELoss()
    rec_criterion = MSELoss()

    encoder = DataParallel(encoder).cuda()
    if args.rec:
        rec_decoder = DataParallel(rec_decoder).cuda()

    total_iters = dataloader_length_max * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        
        encoder.train()
        if args.rec:
            rec_decoder.train()

        tbar = tqdm(zip(*trainloader_list), ncols=150)

        for i, sample_batches in enumerate(tbar):

            img_multi = None
            img_freq_multi = None
            mask_multi = None
            bbox_multi = []
            rec_soft_multi = None
            box_mask_multi = None
            avg_rec_loss = 0.0


            for train_idx in range(len(domain_idx_list)):
                img, img_freq, mask, box= sample_batches[train_idx][0], sample_batches[train_idx][1], \
                                                      sample_batches[train_idx][2], sample_batches[train_idx][3]
                box_mask = create_bitmask(mask,box)
                if img_multi is None:
                    img_multi = img
                    img_freq_multi = img_freq
                    mask_multi = mask
                    box_mask_multi = box_mask
                else:
                    img_multi = torch.cat([img_multi, img], 0)
                    img_freq_multi = torch.cat([img_freq_multi, img_freq], 0)
                    mask_multi = torch.cat([mask_multi, mask], 0)
                    box_mask_multi = torch.cat([box_mask_multi,box_mask], 0)

            img_multi, img_freq_multi, box_mask_multi = img_multi.cuda(), img_freq_multi.cuda(), box_mask_multi.cuda()

            _, pred_hard_1 = encoder(img_multi)

            pred_soft_1 = torch.sigmoid(pred_hard_1)
            loss_bce_1 = compute_project_term(pred_soft_1, box_mask_multi)
            loss_dice_1 = 0 #dice_loss(pred_soft_1, mask_multi)
            
            loss = 0
            if args.ram:
                img_freq_feats, pred_hard_2 = encoder(img_freq_multi)
                img_freq_feats = img_freq_feats['out']
                pred_soft_2 = torch.sigmoid(pred_hard_2)
                loss_bce_2 = compute_project_term(pred_soft_2, box_mask_multi)
                loss_dice_2 = 0 #dice_loss(pred_soft_2, mask_multi)

                if args.consistency:
                    loss_consistency = consistency_criterion(pred_soft_2, pred_soft_1)
                else:
                    loss_consistency = 0
                
                if args.rec:
                    left = 0
                    for train_idx in range(len(domain_idx_list)):
                        right = left + batch_size_list[train_idx]
                        rec_soft = rec_decoder(img_freq_feats[left:right, ...])[1]
                        if rec_soft_multi is None:
                            rec_soft_multi = rec_soft
                        else:
                            rec_soft_multi = torch.cat([rec_soft_multi, rec_soft], 0)
                        loss_rec = rec_criterion(rec_soft, img_multi[left:right])
                        loss = loss + args.lambda_rec * loss_rec
                        avg_rec_loss += loss_rec.item()
                        left = right
            
            else:
                loss_bce_2 = 0
                loss_dice_2 = 0
                loss_consistency = 0

            loss = loss + loss_bce_1 + loss_bce_2 + loss_dice_1 + loss_dice_2 + 0.5 * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            # writer.add_scalar('lr', lr, iter_num)
            # writer.add_scalar('loss/loss_bce_1', loss_bce_1, iter_num)
            # writer.add_scalar('loss/loss_dice_1', loss_dice_1, iter_num)
            # writer.add_scalar('loss/loss_bce_2', loss_bce_2, iter_num)
            # writer.add_scalar('loss/loss_dice_2', loss_dice_2, iter_num)
            # writer.add_scalar('loss/loss_consistency', loss_consistency, iter_num)
            # writer.add_scalar('loss/loss_rec', avg_rec_loss / 4, iter_num)
            #
            # if iter_num  % 100 == 0:
            #     image = img_multi[0:9:4, 0:3, ...]
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     image = img_freq_multi[0:9:4, 0:3, ...]
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image_Freq', grid_image, iter_num)
            #
            #     image = rec_soft_multi[0:9:4, 0:3, ...]
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image_Rec', grid_image, iter_num)
            #
            #     grid_image = make_grid(pred_soft_1[0:9:4, 0, ...].unsqueeze(1), 3, normalize=True)
            #     writer.add_image('train/Soft_Predicted_OC', grid_image, iter_num)
            #
            #     grid_image = make_grid(pred_soft_1[0:9:4, 1, ...].unsqueeze(1), 3, normalize=True)
            #     writer.add_image('train/Soft_Predicted_OD', grid_image, iter_num)
            #
            #     grid_image = make_grid(mask_multi[0:9:4, 0, ...].unsqueeze(1), 3, normalize=False)
            #     writer.add_image('train/GT_OC', grid_image, iter_num)
            #
            #     grid_image = make_grid(mask_multi[0:9:4, 1, ...].unsqueeze(1), 3, normalize=False)
            #     writer.add_image('train/GT_OD', grid_image, iter_num)
            
            iter_num = iter_num + 1

        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            with torch.no_grad():
                avg_dice = test_fundus(encoder, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size, dataset=args.dataset)
            if avg_dice >= previous_best:
                if previous_best != 0:
                    model_path = os.path.join(args.save_path, 'model_%.2f.pth' % (previous_best))
                    if os.path.exists(model_path):
                        os.remove(model_path)
                if args.rec:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                  "rec_decoder_state_dict": rec_decoder.module.state_dict()}
                else:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict()}
                torch.save(checkpoint, os.path.join(args.save_path, 'model_%.2f.pth' % (avg_dice)))
                previous_best = avg_dice
                
    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    if args.rec:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                      "rec_decoder_state_dict": rec_decoder.module.state_dict()}
    else:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict()}
    torch.save(checkpoint, save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))

def train_prostate(trainloader_list, encoder,  rec_decoder, writer, args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list):
    if args.consistency_type == 'mse':
        consistency_criterion = MSELoss()
    elif args.consistency_type == 'kl':
        consistency_criterion = KLDivLoss()
    elif args.consistency_type == 'kd':
        consistency_criterion = KD
    else:
        assert False, args.consistency_type
    criterion = CrossEntropyLoss()
    rec_criterion = MSELoss()

    encoder = DataParallel(encoder).cuda()

    if args.rec:
        rec_decoder = DataParallel(rec_decoder).cuda()

    total_iters = dataloader_length_max * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        
        encoder.train()
        if args.rec:
            rec_decoder.train()

        tbar = tqdm(zip(*trainloader_list), ncols=150)

        for i, sample_batches in enumerate(tbar):

            img_multi = None
            img_freq_multi = None
            mask_multi = None
            bbox_multi = []
            rec_soft_multi = None
            box_mask_multi = None
            avg_rec_loss = 0.0
            for train_idx in range(len(domain_idx_list)):
                img, img_freq, mask,= sample_batches[train_idx][0], sample_batches[train_idx][1], \
                                                      sample_batches[train_idx][2]
                if img_multi is None:
                    img_multi = img
                    img_freq_multi = img_freq
                    mask_multi = mask
                else:
                    img_multi = torch.cat([img_multi, img], 0)
                    img_freq_multi = torch.cat([img_freq_multi, img_freq], 0)
                    mask_multi = torch.cat([mask_multi, mask], 0)

            img_multi, img_freq_multi, mask_multi, = img_multi.cuda(), img_freq_multi.cuda(), mask_multi.cuda()

            img_feats, pred_1 = encoder(img_multi)
            img_feats = img_feats['out']
            pred_soft_1 = torch.softmax(pred_1, dim=1)
            mask_multi = mask_multi.unsqueeze(1)
            loss_ce_1 = compute_project_term(pred_soft_1[:,1:], mask_multi)
            loss_dice_1 = 0 #dice_loss_multi(pred_soft_1, mask_multi, num_classes=args.num_classes, ignore_index=0)

            loss = 0
            if args.ram:
                img_freq_feats, pred_2 = encoder(img_freq_multi)
                img_freq_feats = img_freq_feats['out']
                pred_soft_2 = torch.softmax(pred_2, dim=1)
                loss_ce_2 = compute_project_term(pred_soft_2[:,1:], mask_multi)
                loss_dice_2 = 0 #dice_loss_multi(pred_soft_2, mask_multi, num_classes=args.num_classes, ignore_index=0)
                
                if args.consistency:
                    loss_consistency = consistency_criterion(pred_soft_2, pred_soft_1)
                else:
                    loss_consistency = 0
                
                if args.rec:
                    left = 0
                    for train_idx in range(len(domain_idx_list)):
                        right = left + batch_size_list[train_idx]

                        rec_soft = rec_decoder(img_freq_feats[left:right, ...])[1]
                        if rec_soft_multi is None:
                            rec_soft_multi = rec_soft
                        else:
                            rec_soft_multi = torch.cat([rec_soft_multi, rec_soft], 0)
                        loss_rec = rec_criterion(rec_soft, img_multi[left:right, ...])
                        loss = loss + args.lambda_rec * loss_rec
                        avg_rec_loss += loss_rec.item()
                        left = right
            
            else:
                loss_ce_2 = 0
                loss_dice_2 = 0
                loss_consistency = 0

            loss = loss + loss_ce_1 + loss_ce_2 + loss_dice_1 + loss_dice_2 + 0.5 * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            #
            # writer.add_scalar('lr', lr, iter_num)
            # writer.add_scalar('loss/loss_ce_1', loss_ce_1, iter_num)
            # writer.add_scalar('loss/loss_dice_1', loss_dice_1, iter_num)
            # writer.add_scalar('loss/loss_ce_2', loss_ce_2, iter_num)
            # writer.add_scalar('loss/loss_dice_2', loss_dice_2, iter_num)
            # writer.add_scalar('loss/loss_consistency', loss_consistency, iter_num)
            # writer.add_scalar('loss/loss_rec', avg_rec_loss / 4, iter_num)
            #
            # if iter_num  % 100 == 0:
            #     image = img_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     image = img_freq_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image_Freq', grid_image, iter_num)
            #
            #     image = rec_soft_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
            #     grid_image = make_grid(image, 3, normalize=True)
            #     writer.add_image('train/Image_Rec', grid_image, iter_num)
            #
            #     image = torch.max(pred_soft_1[0:7:3, ...], 1)[1].detach().data.cpu().numpy()
            #     image = decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 3, normalize=False)
            #     writer.add_image('train/Predicted', grid_image, iter_num)
            #
            #     image = mask_multi[0:7:3, ...].detach().data.cpu().numpy()
            #     image = decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 3, normalize=False)
            #     writer.add_image('train/GT', grid_image, iter_num)
            
            iter_num = iter_num + 1
        
        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            with torch.no_grad():
                avg_dice = test_prostate(encoder,  epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size, dataset=args.dataset)
            if avg_dice >= previous_best:
                if previous_best != 0:
                    model_path = os.path.join(args.save_path, 'model_%.2f.pth' % (previous_best))
                    if os.path.exists(model_path):
                        os.remove(model_path)
                if args.rec:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                "rec_decoder_state_dict": rec_decoder.module.state_dict()}
                else:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),}
                torch.save(checkpoint, os.path.join(args.save_path, 'model_%.2f.pth' % (avg_dice)))
                previous_best = avg_dice
                
    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    if args.rec:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                    "rec_decoder_state_dict": rec_decoder.module.state_dict()}
    else:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),}
    torch.save(checkpoint, save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))



def main(args):
    data_root = os.path.join(args.data_root, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if os.path.exists(args.save_path + '/code'):
        shutil.rmtree(args.save_path + '/code')
    shutil.copytree('.', args.save_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))
    
    writer = SummaryWriter(args.save_path + '/log')

    dataset_zoo = {'fundus': Fundus_Multi, 'prostate': Prostate_Multi4}
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.RandomScaleCrop((256, 256))]),
                 'prostate': None}
    batch_size_list = {'fundus': fundus_batch_list[args.test_domain_idx] if args.test_domain_idx < 4 else None,
                       'prostate': prostate_batch_list[args.test_domain_idx]}

    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]

    dataloader_list = []

    dataloader_length_max = -1
    max_id = 0
    max_dataloader = None
    count = 0
    for idx, i in enumerate(domain_idx_list):
        trainset = dataset_zoo[args.dataset](base_dir=data_root, split='train',
                            domain_idx_list=[i], transform=transform[args.dataset], is_out_domain=args.is_out_domain, test_domain_idx=args.test_domain_idx)
        trainloader = DataLoader(trainset, batch_size=batch_size_list[args.dataset][idx], num_workers=8,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)
        dataloader_list.append(cycle(trainloader))
        if dataloader_length_max < len(trainloader):
            dataloader_length_max = len(trainloader)
            max_dataloader = trainloader
            max_id = count
        count += 1
    dataloader_list[max_id] = max_dataloader
    
    encoder = deeplabv3_resnet50(2)
    rec_decoder = PrivateDecoder(2048, 3, False)


    if args.rec:
        if args.dataset == 'fundus':
            optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr / 2},
                            {"params": rec_decoder.parameters(), 'lr': args.lr}],
                            lr=args.lr, betas=(0.9, 0.999))
        else:
            optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr / 2},
                            {"params": rec_decoder.parameters(), 'lr': args.lr}],
                            lr=args.lr, betas=(0.9, 0.999))
    else:
        rec_decoder = None
        optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr}],
                          lr=args.lr, betas=(0.9, 0.999))

    print('\nEncoder Params: %.3fM' % count_params(encoder))
    print('\nRec Decoder Params: %.3fM' % count_params(rec_decoder))
    

    if args.dataset == 'fundus':
        train_fundus(dataloader_list, encoder,  rec_decoder, writer,
                     args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list[args.dataset])
    elif args.dataset == 'prostate':
        train_prostate(dataloader_list, encoder,  rec_decoder, writer,
                       args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list[args.dataset])
    else:
        raise ValueError('Not support Dataset {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:
        args.epochs = {'fundus': 100, 'prostate': 30}[args.dataset]
    if args.lr is None:
        args.lr = {'fundus': 2e-3, 'prostate': 1e-3}[args.dataset]
    if args.num_classes is None:
        args.num_classes = {'fundus': 2, 'prostate': 2}[args.dataset]

    print(args)

    main(args)