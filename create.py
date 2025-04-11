"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
# import os
import time
from collections import OrderedDict

import torch
from torch.backends import cudnn
from torch.autograd import Variable

import cv2
import numpy as np

from src import craft_utils
from src import imgproc
# from src import file_utils

from src.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# """ For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

# result_folder = './result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)



class TextRegions:
    """Provides Text Regions per Image"""

    def __init__(self, **kwargs):
        self.trained_model = kwargs.get("trained_model", 'weights/craft_mlt_25k.pth')
        self.text_threshold = kwargs.get("text_threshold", 0.7)
        self.low_text = kwargs.get("low_text", 0.4)
        self.link_threshold = kwargs.get("link_threshold", 0.4)
        self.device = torch.device(kwargs.get("device", 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.canvas_size = kwargs.get("canvas_size", 1280)
        self.mag_ratio = kwargs.get("mag_ratio", 1.5)
        self.poly = kwargs.get("poly", False)
        self.show_time = kwargs.get("show_time", False)
        self.test_folder = kwargs.get("test_folder", 'data/')
        self.refine = kwargs.get("refine", False)
        self.refiner_model = kwargs.get("refiner_model", 'weights/craft_refiner_CTW1500.pth')

        self.net = CRAFT()

        print('Loading weights from checkpoint (' + self.trained_model + ')')
        self.net.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location=self.device)))

        if self.device == torch.device("cuda"):
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net) # type: ignore
            cudnn.benchmark = False

    def test_net(self, image, show_time=False):
        t0 = time.time()

        # resize
        img_resized, target_ratio, _size_heatmap = imgproc.resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, _ = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if show_time:
            print(f"\ninfer/postproc time : {t0:.3f}/{t1:.3f}")

        return boxes, polys, ret_score_text

    def generateRegions(self, image_path, show_time=False):
        self.net.eval()
        t = time.time()
        image = imgproc.loadImage(image_path)

        _bboxes, polys, _score_text = self.test_net(image, show_time)

        # save score text
        # filename, _ = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)

        # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

        if show_time:
            print(f"elapsed time : {time.time() - t}s")

        return polys
