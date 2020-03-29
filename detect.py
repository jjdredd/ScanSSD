#! /usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from ssd import build_ssd
from data import config


class ArgStub():

    def __init__ (self):
        self.cuda = False
        self.kernel = (1, 5)
        self.padding = (0, 2)
        self.phase = 'test'
        self.visual_threshold = 0.6
        self.verbose = False
        self.exp_name = 'SSD'
        self.model_type = 512
        self.use_char_info = False
        self.limit = -1
        self.cfg = 'hboxes512'
        self.batch_size = 4
        self.num_workers = 2
        self.neg_mining = True
        self.log_dir = 'logs'
        self.stride = 0.1
        self.window = 1200


def draw_box(image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

class MathDetector():

    def __init__(self, weight_path, args):
        net = build_ssd(args, 'test', config.exp_cfg[args.cfg], 0, args.model_type, 2)
        self._net = nn.DataParallel(net)
        weights = torch.load(weight_path, map_location = torch.device('cpu'))
        self._net.load_state_dict(weights)
        self._net.eval()

    def Detect(self, thres, images):

        cls = 1                 # math class
        boxes = []
        scores = []

        y, debug_boxes, debug_scores = self._net(images)  # forward pass
        print('passed')

        detections = y.data

        for k in range(len(images)):

            img_boxes = []
            img_scores = []
            for i in range(detection.size(2)):

                if ( detections[k, cls, j, 0] < thres ):
                    continue

                pt = detections[k, cls, j, 1:]
                coords = (pt[0], pt[1], pt[2], pt[3])
                img_boxes.append(coords)
                img_scores.append(detections[k, cls, j, 0])

            boxes.append(img_boxes)
            scores.append(img_scores)

        return boxes, scores

    def ShowNetwork(self):
        print(self._net)


def get_img():
    img = cv2.imread('/tmp/report_1/report-2.png', cv2.IMREAD_COLOR)
    cimg = img[0:3000, 1000:4000].astype(np.float32)
    rimg = cv2.resize(cimg, (512, 512),
                      interpolation = cv2.INTER_AREA).astype(np.float32)
    rimg -= np.array((246, 246, 246), dtype=np.float32)
    rimg = rimg[:, :, (2, 1, 0)]
    return torch.from_numpy(rimg).permute(2, 0, 1).unsqueeze(0)

md = MathDetector('AMATH512_e1GTDB.pth', ArgStub())
a = get_img()
print(a.device)
