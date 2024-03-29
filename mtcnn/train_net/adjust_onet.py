#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.getcwd())
import mtcnn.train_net.train as train
from mtcnn.core.imagedb import ImageDB

from IPython import embed

anno_dir = '/home/faceu/68keypoints/anno_store'

def train_net(args):

    imagedb    = ImageDB(args.anno_file)
    train_imdb = imagedb.load_imdb()
    # train_imdb = imagedb.append_flipped_images(train_imdb)

    imagedb    = ImageDB(args.eval_file)
    eval_imdb  = imagedb.load_imdb()

    print('train : %d\teval : %d' % (len(train_imdb), len(eval_imdb)))
    print(args)
    train.train_onet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  ONet')

    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'onet/train_anno_48.txt'))
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'onet/eval_anno_48.txt'))
    parser.add_argument('--model_path', type=str,   default='model/onet_v3')
    parser.add_argument('--factors',    type=list,  default=[0.1, 0.7, 8])
    parser.add_argument('--use_lmkinfo',type=bool,  default=True)
    parser.add_argument('--imgsize',    type=int,   default=48)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=3000)
    parser.add_argument('--lr',         type=float, default=1e-3)    # TODO
    parser.add_argument('--batch_size', type=int,   default=64)      # TODO
    parser.add_argument('--use_cuda',   type=bool,  default=True)    # TODO
    parser.add_argument('--prefix_path',type=str,   default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
