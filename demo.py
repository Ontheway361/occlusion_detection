#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
from mtcnn.core.vision import visual_face
from mtcnn.core.detect import MtcnnDetector

from IPython import embed


def parse_args():

    parser = argparse.ArgumentParser(description='Inference of MTCNN')
    
    parser.add_argument('--pnet_file',  type=str,  default='model/pnet.pt')
    parser.add_argument('--rnet_file',  type=str,  default='model/rnet.pt')
#     parser.add_argument('--onet_file',  type=str,  default='model/onet.pt')
#     parser.add_argument('--onet_file',  type=str,  default='model/onet_v1/onet_epoch_7.pt')
#     parser.add_argument('--onet_file',  type=str,  default='model/onet_v2/onet_epoch_6.pt')
    parser.add_argument('--onet_file',  type=str,  default='model/onet_v3/onet_epoch_10.pt')
    parser.add_argument('--use_cuda',   type=bool, default=True)   # TODO
    parser.add_argument('--gpu_ids',    type=list, default=[0, 1]) # TODO
    parser.add_argument('--prob_thres', type=list, default=[0.6, 0.7, 0.7])

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    imglists = [s.split('.')[0] for s in os.listdir('aku_imgs/')]
    mtcnn_detector = MtcnnDetector(parse_args())
    
    for img_name in imglists:
        
        try:
            img = cv2.imread('aku_imgs/%s.jpg' % img_name)
            bboxs, landmarks = mtcnn_detector.detect_face(img, verbose=False)
            save_name = 'result/r3_%s.jpg' % img_name
            print('save img name : %s' % save_name)
            visual_face(img, bboxs, landmarks, save_name)
        except Exception as e:
            print(e)
