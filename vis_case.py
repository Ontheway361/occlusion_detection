#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/11/06
author: relu
'''


import os
import cv2
import argparse
from PIL import Image, ImageDraw
from mtcnn.core.detect import MtcnnDetector

from IPython import embed

root_dir = '/Users/relu/data/deep_learning/face_landmark/imgs_occ/'    # mini-mac
# root_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/'                   # gpu3


class VisSingleCase(object):

    def __init__(self, args):

        self.model = MtcnnDetector(args)


    def _fliter_doc_bbox(self, bboxes, landmarks):
        ''' Filter the face_box on card '''

        area = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1) * -1
        area_index = area.argsort()
        bbox = bboxes[area_index[0]]
        landmark = landmarks[area_index[0]]
        #prob_index = (bboxes[-1]*-1).argsort() # assist
#         if bboxes.shape[0] == 2 or area_index[0] == prob_index[0]:
#             bbox = bboxes[area_index[0]]
#             landmark = landmark[area_index[0]]

        return bbox, landmark


    def _fetch_block(self, bbox, landmark, lmk_flag = True):
        ''' Crop the chin_block of the detected face '''

        if lmk_flag:
            landmark = landmark.reshape(-1,2)
            left_down  = landmark[6]  # default : 6
            nose_point = landmark[33] # point-34 | nose
            right_down = landmark[10] # default : 10
            chin_point = landmark[8]  # point-9  | chin

            x1, y1 = int(left_down[0]), int(nose_point[1])
            x2, y2 = int(right_down[0]), int(chin_point[1])
        else:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

        return (x1, y1, x2, y2)


    def _vis_result(self, img, bbox, landmark):
        ''' Visual the detect-result and block '''

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)
        block   = self._fetch_block(bbox, landmark)

        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='yellow')
        draw.rectangle([(block[0], block[1]), (block[2], block[3])], outline='red')

        pts = landmark.reshape((68, 2))
        for idx in range(68):
            point = (pts[idx,0]-1, pts[idx,1]-1, pts[idx,0]+1, pts[idx,1]+1)
            draw.ellipse(point, fill=None, outline='green')
        pil_img.show()


    def check_case(self, img_path):
        ''' Check the single case '''

        try:
            img = cv2.imread(img_path)
            bboxes, landmarks = self.model.detect_face(img, verbose=False)
            print((img_path, bboxes.shape))
        except Exception as e:
            print(e)
        else:
            if bboxes.shape[0] == 0:
                print('No face detected in %s' % img_path)
            else:
                bbox, landmark = self._fliter_doc_bbox(bboxes, landmarks)
                self._vis_result(img, bbox, landmark)


def parse_args():

    parser = argparse.ArgumentParser(description='Inference of MTCNN')

    # common
    parser.add_argument('--pnet_file', type=str, default='model/68-pts/pnet.pt')
    parser.add_argument('--rnet_file', type=str, default='model/68-pts/rnet.pt')
    parser.add_argument('--onet_file', type=str, default='model/68-pts/onet_v1.pt')  # onet_v1.pt
    parser.add_argument('--prob_thres',type=list,default=[0.6, 0.7, 0.7])  # CORE
    parser.add_argument('--use_cuda',  type=bool,default=False)   # TODO
    parser.add_argument('--gpu_ids',   type=list,default=[0, 1]) # TODO

    # path-dir
    parser.add_argument('--data_dir',  type=str, default=root_dir)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    vis_engine = VisSingleCase(args)
    # miss_imgs  = [18, 52, 60, 63, 68, 75, 77]
    # error_imgs = [9, 23, 33, 34, 39]
    check_imgs = [9]
    for idx in check_imgs:

        img_path = os.path.join(args.data_dir, 'faces_1031/%d.jpg' % idx)
        vis_engine.check_case(img_path)
