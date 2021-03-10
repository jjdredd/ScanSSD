#! /usr/bin/env python3

import os
from collections import OrderedDict
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



def draw_box (image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)



def _img_to_tensor (image):
    rimg = cv2.resize(image, (512, 512),
                      interpolation = cv2.INTER_AREA).astype(np.float32)
    rimg -= np.array((246, 246, 246), dtype=np.float32)
    rimg = rimg[:, :, (2, 1, 0)]
    return torch.from_numpy(rimg).permute(2, 0, 1)



def FixImgCoordinates (images, boxes):
    new_boxes = []
    if isinstance(images, list):
            for i in range(len(images)):
                print(images[i].shape)
                bbs = []
                for o_box in boxes[i] :
                    b = [None] * 4
                    b[0] = int(o_box[0] * images[i].shape[0])
                    b[1] = int(o_box[1] * images[i].shape[1])
                    b[2] = int(o_box[2] * images[i].shape[0])
                    b[3] = int(o_box[3] * images[i].shape[1])
                    bbs.append(b)

                new_boxes.append(bbs)
    else:
        bbs = []
        for o_box in boxes[0] :
            b = [None] * 4
            b[0] = int(o_box[0] * images.shape[0])
            b[1] = int(o_box[1] * images.shape[1])
            b[2] = int(o_box[2] * images.shape[0])
            b[3] = int(o_box[3] * images.shape[1])
            bbs.append(b)

            # this could be
            # b[0] = int(o_box[0] * images.shape[0]) ==> b[0] = int(o_box[0] * images.shape[1])
            # b[1] = int(o_box[1] * images.shape[1]) ==> b[1] = int(o_box[1] * images.shape[0])
            # b[2] = int(o_box[2] * images.shape[0]) ==> b[2] = int(o_box[2] * images.shape[1])
            # b[3] = int(o_box[3] * images.shape[1]) ==> b[3] = int(o_box[3] * images.shape[0])

        new_boxes.append(bbs)

    return new_boxes


def DrawAllBoxes(images, boxes):
    for i in range(len(images)):
        draw_box(images[i], boxes[i])

class MathDetector():

    def __init__(self, weight_path, args):
        net = build_ssd(args, 'test', config.exp_cfg[args.cfg], 0, args.model_type, 2)
        self._net = net # nn.DataParallel(net)
        weights = torch.load(weight_path, map_location = torch.device('cpu'))

        new_weights = OrderedDict()
        for k, v in weights.items():
            name = k[7:] # remove `module.`
            new_weights[name] = v

        self._net.load_state_dict(new_weights)
        self._net.eval()

    def Detect (self, thres, images):

        cls = 1                 # math class
        boxes = []
        scores = []

        y, debug_boxes, debug_scores = self._net(images)  # forward pass

        detections = y.data

        for k in range(len(images)):

            img_boxes = []
            img_scores = []
            for j in range(detections.size(2)):

                if ( detections[k, cls, j, 0] < thres ):
                    continue

                pt = detections[k, cls, j, 1:]
                coords = (pt[0], pt[1], pt[2], pt[3])
                img_boxes.append(coords)
                img_scores.append(detections[k, cls, j, 0])

            boxes.append(img_boxes)
            scores.append(img_scores)

        return boxes, scores

    def ShowNetwork (self):
        print(self._net)

    def DetectAny (self, thres, image):
        if isinstance(image, list):
            t_list = [_img_to_tensor(img) for img in image]
            t = torch.stack(t_list, dim = 0)
        else:
            t = _img_to_tensor(image).unsqueeze(0)
        # fix box coordinates to image pixel coordinates
        boxes, scores = self.Detect(thres, t)
        return FixImgCoordinates(image, boxes), scores


def get_img():
    img = cv2.imread('/tmp/report-2.png', cv2.IMREAD_COLOR)
    cimg = img[0:3000, 1000:4000].astype(np.float32)
    return cimg

md = MathDetector('AMATH512_e1GTDB.pth', ArgStub())
# a = get_img()

# a = cv2.imread('/tmp/s-48.png', cv2.IMREAD_COLOR)
exit(0)

b, s = md.DetectAny(0.2, a)

print(s)

DrawAllBoxes([a, ], b)
cv2.imwrite('/tmp/1.png', a)


# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

# # original saved file with DataParallel
# state_dict = torch.load('myfile.pth.tar')
# # create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)
