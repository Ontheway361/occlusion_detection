#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
import mtcnn.core.utils as utils
from mtcnn.core.detect import MtcnnDetector
from mtcnn.core.vision import visual_face, visual_prob

from IPython import embed


class HistOccBlock(object):
    
    def __init__(self, args):
        
        self.args      = args
        self.model     = None
        self.imgs_list = None
        self.bgr_prob  = np.zeros((3, 256), dtype=np.float)
    
    
    def _prepare(self):
        ''' Prepare the face-model, imgs_list '''
        
        # face_model
        self.model = MtcnnDetector(self.args)
        
        # imgs_list
        folder_path = os.path.join(self.args.data_dir, self.args.folder)
        imgs_list   = []
        for folder in os.listdir(folder_path):
            
            idx = folder.split('_')[-1]
            img_name = '%s_%d.jpg' % (self.args.img_type, int(idx))
            imgs_list.append(os.path.join(folder_path, folder, img_name))
        self.imgs_list = imgs_list
   
    
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
        

    def _fetch_block(self, img, landmark):
        ''' Crop the chin_block of the detected face '''
        
        height, width, _ = img.shape
        landmark = landmark.reshape(-1,2)
        left_down  = landmark[6]
        nose_point = landmark[33] # point-34 | nose
        right_down = landmark[10]
        chin_point = landmark[8]  # point-9  | chin
        
        x1, y1 = int(left_down[0]), int(nose_point[1])
        x2, y2 = int(right_down[0]), int(chin_point[1])
        crop_block = img[y1:y2 + 1, x1:x2 + 1, :]
        
        return crop_block
        
    
    
    def _statistics(self, block):
        ''' Statistics the block-pixels info '''
        
        height, width, _ = block.shape
        num_pixels = height * width
        bgr_prob   = np.zeros((3, 256), dtype=np.int)
        
        for y in range(height):
            for x in range(width):
                bgr_prob[:, block[y,x]] += 1
        bgr_prob = bgr_prob / num_pixels
        
        return bgr_prob
   
    
    def _hist_go(self):
        ''' 
        Statistic the info of resz_block 
        step - 1. detect face with trained model
        step - 2. filter the doc_bbox
        step - 3. crop and resize the target block
        step - 4. get the statistics_info
        '''
        
        face_count = 1;  # skip zero
        for img_path in self.imgs_list:
            
            try:
                img = cv2.imread(img_path)
                bboxes, landmarks = self.model.detect_face(img, verbose=False)
                save_name = 'result/r_%s' % img_path.split('/')[-1]
                visual_face(img, bboxes, landmarks, save_name)
                print((img_path, bboxes.shape))
            except Exception as e:
                print(e)
            else:
                if bboxes.shape[0] == 0:
                    print('No face detected in %s' % img_path)
                    continue
                else:
                    bbox, landmark = self._fliter_doc_bbox(bboxes, landmarks)
                face_count += 1
                block = self._fetch_block(img, landmark)
                self.bgr_prob += self._statistics(block)
        self.bgr_prob /= face_count
                
                
    def runner(self):
        ''' Pipeline of HistOccBlock '''
        
        self._prepare()
        self._hist_go()
        embed()
        visual_prob(self.bgr_prob, '.pdf/', self.args.num_bins)
        
        return hist_dict
         


def parse_args():

    parser = argparse.ArgumentParser(description='Inference of MTCNN')
    
    parser.add_argument('--pnet_file', type=str, default='model/pnet.pt')
    parser.add_argument('--rnet_file', type=str, default='model/rnet.pt')
    parser.add_argument('--onet_file', type=str, default='model/onet.pt')  
    parser.add_argument('--prob_thres',type=list,default=[0.6, 0.7, 0.7])  # CORE
    parser.add_argument('--use_cuda',  type=bool,default=True)   # TODO
    parser.add_argument('--gpu_ids',   type=list,default=[0, 1]) # TODO
    parser.add_argument('--data_dir',  type=str, default='/home/jovyan/gpu3-data2/lujie/imgs_occ/') 
    parser.add_argument('--folder',    type=str, default='occ_pair')  # {occ_pair, consist_pair, wrong_pair}
    parser.add_argument('--img_type',  type=str, default='input')     # {source, input} 
    parser.add_argument('--num_bins',  type=int, default=32)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    hist_engine = HistOccBlock(parse_args())
    hist_engine.runner()

    
