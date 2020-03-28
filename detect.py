#! /usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ssd import build_ssd
from data import config

def draw_box(image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

class MathDetector():

    def __init__(self, weight_path):
        net = build_ssd(args, 'test', config['gtdb'], 0, 300, 2)
        self._net = nn.DataParallel(net)
        self._net.load_state_dict(torch.load(weight_path))
        sel._net.eval()
