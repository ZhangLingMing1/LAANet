import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import plydataset
from utils import test_semseg
from model import LAANet


if __name__ == "__main__":
    gpu = False
    path = "lastest.pth"
    testall_dataset = plydataset(path="dental_data/input")
    testall_dataloader = DataLoader(dataset=testall_dataset, batch_size=1, num_workers=1, shuffle=True)
    print('load model %s' % path)
    model = LAANet(in_channel=21, num_classes=8)
    if gpu:
       model.cuda()
       checkpoint = torch.load(path)
    else:
       checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    test_metrics, test_hist_acc, cat_mean_iou = test_semseg(model, testall_dataloader, num_classes=8,
                                                                gpu=gpu, generate_ply=True)
    print(test_metrics)
