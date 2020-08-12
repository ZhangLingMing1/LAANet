import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from time import time



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids





def KNN(nsample, xyz, new_xyz, fps_idx):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_id = torch.zeros((B, S, nsample)).long()  # 每个Local区域，邻居点的索引  [B, S, nsample]
    for i in range(S):
        centre = new_xyz[:, i, :].reshape(B, 1, 3)
        distance = (xyz - centre) ** 2
        distance = (distance.sum(2)) ** (1 / 2)  # 中心点和其他点的距离 [B, N]
        _, sort_idx = distance.sort()
        for j in range(B):
            group_id[j, i, :] = sort_idx[j, 1:nsample+1]  # 最近的点为自己，所以从1开始取
    return group_id



def sample_and_group(npoint, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = KNN(nsample, xyz, new_xyz, fps_idx)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    #grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        new_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        #fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


class Position_Augmentation(nn.Module):

    def __init__(self, out_channels):
        super(Position_Augmentation, self).__init__()
        self.out_channels = out_channels
        self.mlp_aug = nn.Conv2d(in_channels=10, out_channels=self.out_channels, kernel_size=1)
        self.bn_aug = nn.BatchNorm2d(self.out_channels)

    def forward(self, new_xyz, new_points, grouped_xyz, fps_points):
        B, S, n, C = grouped_xyz.shape
        '''计算邻点与中心点的距离'''
        grouped_xyz_norm = (grouped_xyz - new_xyz.view(B, S, 1, C)) ** 2
        grouped_xyz_distance = (grouped_xyz_norm.sum(-1)) ** 0.5
        grouped_xyz_distance = grouped_xyz_distance.view(B, S, n, 1)  # [B, npoint, nsample, 1]
        aug = torch.cat(
            [new_xyz.view(B, S, 1, C).repeat(1, 1, n, 1), grouped_xyz, grouped_xyz_norm,
             grouped_xyz_distance],
            dim=-1)  # [B, npoint, nsample, 10]
        aug = aug.permute(0, 3, 2, 1)  # [B, 10, nsample, npoint]
        '''增强后的相对位置信息送入MLP进行学习'''
        aug = F.relu(self.bn_aug(self.mlp_aug(aug)))  # [B, self.added_in_channel, nsample, npoint]
        '''将每个邻点学习到的更丰富位置信息与其对应的特征信息进行Concat'''
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]
        new_points = torch.cat([new_points, aug], dim=1)  # [B, C+self.added_in_channel, nsample, npoint]
        '''将邻点的位置信息进行MaxPooling得到的结果和中心点的特征信息进行Concat'''
        max_out = torch.max(aug, dim=2)[0].view(B, -1, 1, S)  # [B, self.added_in_channel, 1, npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1)  # [B, C, 1, npoint]
        fps_points = torch.cat([fps_points, max_out], dim=1)  # [B, C+self.added_in_channel, 1, npoint]
        return new_points, fps_points, grouped_xyz_distance

class Local_Attention_Aggregate(nn.Module):
    def __init__(self, in_channel, feature_dim):
        super(Local_Attention_Aggregate, self).__init__()
        self.mlp = nn.Conv2d(in_channel, feature_dim, 1)
        self.bn = nn.BatchNorm2d(feature_dim)


    def forward(self,  new_points, fps_points, new_xyz, grouped_xyz, grouped_xyz_distance):

        B, npoint, _ = new_xyz.size()
        _, _, nsample,C = fps_points.size()
        delta_f = fps_points - new_points.view(B, npoint, 1, C).expand(B, npoint, nsample, C)
        r_lo = torch.cat([new_xyz.view(B, npoint, 1, 3).repeat(1, 1, nsample, 1), grouped_xyz,
                          grouped_xyz_distance,delta_f],dim=-1).permute(0, 3, 2, 1)
        weight = F.relu(self.bn(self.mlp(r_lo))).permute(0, 3, 2, 1) # [B, npoint, nsample,D]
        attention = F.softmax(weight, dim=2) # [B, npoint, nsample,D]
        #attention = F.dropout(attention, 0.6, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, fps_points),dim = 2) # [B, npoint, D]
        new_points = new_points + graph_pooling
        return new_points


class Local_Feature_Extrection(nn.Module):
    def __init__(self, npoint,nsample, in_channel, mlp ,droupout=0.6, alpha=0.2):
        super(Local_Feature_Extrection, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        self.added_in_channel = 32
        self.position_aug = Position_Augmentation(out_channels=self.added_in_channel)
        last_channel = in_channel + self.added_in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.LAA = Local_Attention_Aggregate(in_channel=7+last_channel, feature_dim=last_channel)


    def forward(self, xyz, points):

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)


        new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint,
                                                                            self.nsample, xyz, points, True)
        ''' new_xyz: sampled points position data, [B, npoint, 3]
            fps_points: [B, npoint, C]
            grouped_xyz: [B, npoint, nsample, 3]
            new_points: sampled points data, [B, npoint, nsample, C]
        '''
        B, S, C = new_xyz.shape


        new_points, fps_points, grouped_xyz_distance = self.position_aug(new_xyz, new_points, grouped_xyz, fps_points)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points = F.relu(bn(conv(new_points)))

        new_points = self.LAA(new_xyz=new_xyz,
                              new_points=fps_points.squeeze(2).permute(0, 2, 1),
                              grouped_xyz=grouped_xyz,
                              fps_points=new_points.permute(0, 3, 2, 1),
                              grouped_xyz_distance = grouped_xyz_distance
                              )

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module): 
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class LAANet(nn.Module):
    def __init__(self, num_classes=8, in_channel=9, droupout=0, alpha=0.2):
        super(LAANet, self).__init__()

        self.LFE1 = Local_Feature_Extrection(4000, 16, in_channel, [64, 64, 64], droupout,alpha)
        self.LFE2 = Local_Feature_Extrection(2000, 16, 64, [128, 128, 128], droupout,alpha)
        self.LFE3 = Local_Feature_Extrection(1000, 16, 128, [256, 256, 256], droupout,alpha)
        self.LFE4 = Local_Feature_Extrection(500, 16, 256, [512, 512, 512], droupout,alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(droupout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, point):
        l1_xyz, l1_points = self.LFE1(xyz, point)
        l2_xyz, l2_points = self.LFE2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.LFE3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.LFE4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import os
    import torch
    xyz = torch.randn((1, 3, 16000))
    points = torch.randn((1, 21, 16000))
    model = LAANet(num_classes=8, in_channel=21)
    x = model(xyz, points)




