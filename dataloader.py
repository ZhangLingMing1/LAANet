from plyfile import PlyData
import numpy as np
import random
from torch.utils.data import DataLoader,Dataset,random_split
import os
import torchvision.transforms as transforms
import torch
import math
import pandas as pd
import time
# 八种label对应的RGB值
labels = ((255, 255, 255), (255, 0, 0), (255, 125, 0),(255, 255, 0), (0, 255, 0), (0, 255, 255),
          (0, 0, 255), (255, 0, 255))

def get_data(path=""):
    """
    Input:
        path: path of ply file
    Return:
        points: coordinate of points [N, 3]
        label_points: label of points [N, 1]
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
    """
    row_data = PlyData.read(path)  # 读ply文件
    vertex = row_data.elements[0].data  # 点坐标+RGB+alpha信息
    face = row_data.elements[1].data  # face的点索引+face的RGB标签
    n_point = vertex.shape[0]  # 顶点个数
    n_face = face.shape[0]  # 网格个数
    points = np.zeros([n_point, 6])  # 点云坐标+法向量
    index_face = np.zeros([n_face, 3]).astype('int')  # 组成网格的点的索引
    points_face = np.zeros([n_face, 21]).astype('float32')  # 3个点坐标+3个点法向量+中心点坐标
    label_face = np.zeros([n_face, 1]).astype('int64')  # face标签
    normal_face = np.zeros([n_face,3]).astype('float32')  # face的法向量

    for i, data in enumerate(vertex):
        # get coordinate and normal of points
        points[i][:6] = [data[0], data[1], data[2], data[3], data[4], data[5]]

    for i, data in enumerate(face):
        index_face[i, :] = [data[0][0], data[0][1], data[0][2]]  # get index of points
        # get coordinate of  3 point of face
        points_face[i, :3] = points[data[0][0], :3]
        points_face[i, 3:6] = points[data[0][1], :3]
        points_face[i, 6:9] = points[data[0][2], :3]
        points_face[i, 9:12] = points[data[0][0], 3:]
        points_face[i, 12:15] = points[data[0][1], 3:]
        points_face[i, 15:18] = points[data[0][2], 3:]
        # get center point of face
        points_face[i, 18] = (points[data[0][0], 0] + points[data[0][1], 0] + points[data[0][2], 0]) / 3
        points_face[i, 19] = (points[data[0][0], 1] + points[data[0][1], 1] + points[data[0][2], 1]) / 3
        points_face[i, 20] = (points[data[0][0], 2] + points[data[0][1], 2] + points[data[0][2], 2]) / 3
        # get normal of each face
        x1, y1, z1 = points_face[i, :3]
        x2, y2, z2 = points_face[i, 3:6]
        x3, y3, z3 = points_face[i, 6:9]
        normal_face[i, 0] = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
        normal_face[i, 1] = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
        normal_face[i, 2] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        # get label of face
        R, G, B = data[1], data[2], data[3]
        for j, label in enumerate(labels):
            if R == label[0] and G == label[1] and B == label[2]:
                label_face[i] = j
                break
    return points, index_face, points_face, label_face, normal_face

def get_data_v2(path=""):
    labels = ([255, 0, 0], [255, 125, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
              [0, 0, 255], [255, 0, 255])
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    n_face = faces.shape[0]  # number of faces
    xyz = points[:, :3] # coordinate of vertex shape=[N, 3]
    normal = points[:, 3:]  # normal of vertex shape=[N, 3]
    label_face = np.zeros([n_face,1]).astype('int32')
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    """ RGB of faces shape=[N, 3] """
    RGB_face = faces[:, 1:4]
    """ coordinate of 3 vertexes  shape=[N, 9] """
    xyz_face = np.concatenate((xyz[index_face[:, 0], :], xyz[index_face[:, 1], :],xyz[index_face[:, 2], :]), axis=1)
    """  normal of 3 vertexes  shape=[N, 9] """
    normal_vertex = np.concatenate((normal[index_face[:, 0], :], normal[index_face[:, 1], :],normal[index_face[:, 2], :]), axis=1)
    """ get normal and centre of each faces, shape=[N, 3]"""
    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    normal_face1 = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    normal_face2 = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    normal_face3 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    normal_face = np.concatenate((normal_face1, normal_face2, normal_face3), axis=0).reshape(n_face, 3)
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate((x_centre.reshape(n_face,1),y_centre.reshape(n_face,1),z_centre.reshape(n_face,1)), axis=1)
    """ get points of each face, concat all of above"""
    points_face = np.concatenate((xyz_face, normal_vertex, normal_face, centre_face), axis=1)
    """ get label of each face """
    for i, label in enumerate(labels):
        label_face[(RGB_face == label).all(axis=1)] = i+1
    return index_face, points_face, label_face

def generate_plyfile(index_face, point_face, label_face, path= " "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    flag = np.zeros([unique_index.max()+1, 2]).astype('uint64')
    order = 0
    with open(path, "a") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i, index in enumerate(index_face):
            for j, data in enumerate(index):
                if flag[data, 0] == 0:  # if this point has not been wrote
                    xyz = point_face[i, 3*j:3*(j+1)]  # Get coordinate
                    xyz_nor = point_face[i, 3*(j+3):3*(j+4)]
                    f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                            + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + "\n")
                    flag[data, 0] = 1  # this point has been wrote
                    flag[data, 1] = order  # give point a new index
                    order = order + 1  # index add 1 for next point

        for i, data in enumerate(index_face):  # write new point index for every face
            RGB = labels[label_face[i, 0]]  # Get RGB value according to face label
            f.write(str(3) + " " + str(int(flag[data[0], 1])) + " " + str(int(flag[data[1], 1])) + " "
                    + str(int(flag[data[2], 1])) + " " + str(RGB[0]) + " " + str(RGB[1]) + " "
                    + str(RGB[2]) + " " + str(255) + "\n")
        f.close()

class plydataset(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """
    def __init__(self, path="row_data/train"):
        self.root_path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_face, label_face = get_data_v2(path=read_path)
        max_coordinate = abs(max(points_face[:, :9].max(), points_face[:, :9].min(), key=abs))
        points_face[:, :9] = points_face[:, :9]/max_coordinate  # normalize coordinate to (-1,1)
        return index_face, points_face, label_face, self.file_list[item]







