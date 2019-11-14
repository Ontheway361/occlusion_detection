#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/11/04
author: relu
'''

import os
import argparse
import utils.occ_utils as occ_utils
import utils.xgb_classifier as xgb_classifier
from mtcnn.core.vision import visual_face, visual_prob

# root_dir = '/Users/relu/data/deep_learning/face_landmark/imgs_occ/'    # mini-mac
root_dir = '/Users/relu/data/deep_learning/face_landmark/aku_dataset_occ/filter_data_1113'
# root_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/'                   # gpu3

def parse_args():

    parser = argparse.ArgumentParser(description='Inference of MTCNN')

    # common
    parser.add_argument('--pnet_file', type=str, default='model/68-pts/pnet.pt')
    parser.add_argument('--rnet_file', type=str, default='model/68-pts/rnet.pt')
    parser.add_argument('--onet_file', type=str, default='model/68-pts/onet_v1.pt')
    parser.add_argument('--prob_thres',type=list,default=[0.6, 0.7, 0.7])  # CORE
    parser.add_argument('--use_cuda',  type=bool,default=False)   # TODO
    parser.add_argument('--gpu_ids',   type=list,default=[0, 1]) # TODO

    # path-dir
    parser.add_argument('--data_dir',  type=str, default=root_dir)
    parser.add_argument('--csv_file',  type=str, default='face_occ_1111.csv')  # [face_occ_1111]
    parser.add_argument('--folder',    type=str, default='real_aku_pos')
    parser.add_argument('--check_mode',type=str, default='test') # [pos, neg, test]

    # eval
    parser.add_argument('--num_bins',  type=int,  default=256)     # TODO
    parser.add_argument('--anno_date', type=str,  default='1031')
    parser.add_argument('--order',     type=int,  default=1)
    parser.add_argument('--rre_thres', type=float,default=-0.01)
    parser.add_argument('--xgb_thres', type=float,default=0.75)  # TODO
    parser.add_argument('--c_weights', type=str,  default=[1, 1, 1])
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # hist_engine = occ_utils.HistOccBlock(parse_args())
    # hist_engine.runner(vis=False)

    eval_engine = occ_utils.OcclusionEval(parse_args())
    eval_engine.runner(vis=False)

    # xgb_engine = xgb_classifier.XGBClassifier(parse_args())
    # xgb_engine.runner(num_exps=1, reproduce=False)
