import os

import cv2
import numpy as np
from PIL import Image

from textblockdetector.textblock import visualize_textblocks
from textblockdetector import run, TextDetector, REFINEMASK_INPAINT
from textblockdetector.utils.yolov5_utils import non_max_suppression

model = TextDetector(model_path='comictextdetector.pt.onnx', act='leaky', input_size=1024)


def preprocessing(image):
    img = Image.open(image).convert('RGB')
    img = np.array(img)
    img_in, ratio, dw, dh, im_h, im_w = model.preprocessing(img)
    return img_in.numpy(), im_h, im_w, dw, dh, img


def postprocessing(im_h, im_w, dw, dh, img, blks1, blks2, mask, lines_map):
    img = Image.open(img).convert('RGB')
    img = np.array(img)
    blks = np.concatenate((blks1,blks2), axis=1)
    mask, final_mask, textlines = model.postprocessing(im_h, im_w, dw, dh, img, blks, mask, lines_map,
                                                       refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False,
                                                       bgr2rgb=False)

    for i in textlines:
        print(i.xyxy)

