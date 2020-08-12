# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
from dataloader import generate_plyfile, plydataset

def compute_cat_iou(pred,target,iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]  # batch_pred [N,C]
        batch_target = target[j]  # batch_target [N]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()  # index of max value  batch_choice [N]
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_semseg(model, loader, num_classes = 8, gpu=False, generate_ply=False):
    '''
    Input
    :param model:
    :param loader:
    :param num_classes:
    :param pointnet2:
    Output
    metrics: metrics['accuracy']-> overall accuracy
             metrics['iou']-> mean Iou
    hist_acc: history of accuracy
    cat_iou: IoU for o category
    '''
    iou_tabel = np.zeros((8,3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (index, points, target, name) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points_face = points[0].numpy()
        index_face = index[0].numpy()
        xyz = points[:, :, 21:]
        points = points[:, :, 0:21]
        points = points.transpose(2, 1)
        xyz = xyz.transpose(2, 1)
        target = target[:, :, 0]
        points, target = Variable(points.float()), Variable(target.long())
        if gpu:
            xyz, points, target = xyz.cuda(), points.cuda(), target.cuda()
        with torch.no_grad():
               pred = model(xyz, points)

        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        label_face = pred_choice.cpu().reshape(pred_choice.shape[0], 1)
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
        if generate_ply:
               generate_plyfile(index_face, points_face, label_face, path=("dental_data/prediction/%s") % name)
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    if iou_tabel[1,1] == 0:#no label1
        metrics['iou'] = np.mean(np.concatenate((iou_tabel[0:1, 2],iou_tabel[2:, 2])))
    else:
        metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = ["label%d"%(i) for i in range(8)]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou








