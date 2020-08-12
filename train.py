import argparse
import torch
import numpy as np
import torch.nn.parallel
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import logging
import os
from pathlib import Path
from dataloader import plydataset, generate_plyfile
from utils import test_semseg
from tqdm import tqdm
from model import LAANet

from utils import compute_cat_iou
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser('Local_Attention_Aggregate')
    parser.add_argument('--batchsize', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=201, help='number of epochs for training')
    parser.add_argument('--gpu', type=bool, default=False, help='whether use GPU')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    return parser.parse_args()


def main(args):
    print("begin")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    '''-----Create Folder-----'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/LAANet' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''-------LOG------'''
    logger = logging.getLogger("LAANet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_LAANet.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('github')
    '''-------Load Data------'''
    print("Load data")
    train_dataset = plydataset(path="dental_data/input")
    train_Loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=1)
    test_dataset = plydataset(path="dental_data/input")
    test_Loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True, num_workers=1)

    '''------build network------'''
    model = LAANet(in_channel=21, num_classes=8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if args.gpu:
        model.cuda()
    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0
    LEARNING_RATE_CLIP = 1e-4  # lower limit of learning rate
    for epoch in range(0, args.epoch):
        scheduler.step()  # only run this word, the lr can be undated
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)  # avoiding lr is too small
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # set lr

        for i, data in tqdm(enumerate(train_Loader, 0),total=len(train_Loader), smoothing=0.9):
            index_face, point_face, label_face, _ = data
            xyz = point_face[:, :, 21:]
            point_face = point_face[:, :, 0:21]
            xyz = Variable(xyz.float())
            point_face,  label_face = Variable(point_face.float()), Variable(label_face.long())
            point_face = point_face.transpose(2, 1)
            xyz = xyz.transpose(2, 1)
            if args.gpu:
                xyz = xyz.cuda()
                point_face = point_face.cuda()
                label_face = label_face.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred= model(xyz, point_face)
            pred = pred.contiguous().view(-1, 8)  # shape must be [BxN, 8]
            label_face = label_face.view(-1, 1)[:, 0]  # shape must be [BxN]
            loss = F.nll_loss(pred, label_face)

            print("loss=%f" % loss.cpu().data.numpy())
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), '%s/lastest.pth' % (checkpoints_dir))
        if epoch % 2 == 0:
            test_metrics, test_hist_acc, cat_mean_iou = test_semseg(model, test_Loader,
                                                                num_classes=8, gpu=args.gpu)
            mean_iou = np.mean(cat_mean_iou)
            print('Epoch %d   accuracy: %f  meanIOU: %f' % (epoch, test_metrics['accuracy'], mean_iou))
            logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (epoch, 'test', test_metrics['accuracy'], mean_iou))
            if test_metrics['accuracy'] > best_acc:
               best_acc = test_metrics['accuracy']
               torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (checkpoints_dir,"LAANet", epoch, best_acc))
               logger.info(cat_mean_iou)
               logger.info('Save model..')
               print('Save model..')
               print(cat_mean_iou)
            if mean_iou > best_meaniou:
               best_meaniou = mean_iou
               print('Best accuracy is: %.5f' % best_acc)
               logger.info('Best accuracy is: %.5f' % best_acc)
               print('Best meanIOU is: %.5f' % best_meaniou)
               logger.info('Best meanIOU is: %.5f' % best_meaniou)




if __name__ == "__main__":
    args = parse_args()
    main(args)
