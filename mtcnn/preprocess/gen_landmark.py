#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    Design for 300W::68-keypoints landmark dataset
'''

import os
import cv2
import sys
import time
import random
import argparse
import numpy as np
sys.path.append(os.getcwd())
import mtcnn.core.utils as utils
import mtcnn.core.landmark_utils as lmk_utils

from IPython import embed

root_dir = '/home/faceu'

def gen_data(args):

    if args.img_size == 12:
        folder_name = 'pnet'
    elif args.img_size == 24:
        folder_name = 'rnet'
    elif args.img_size == 48:
        folder_name = 'onet'
    else:
        raise TypeError('image_size must be 12, 24 or 48')

    txt_save_dir = os.path.join(args.save_dir, 'anno_store/%s' % folder_name)
    img_save_dir = os.path.join(args.save_dir, '%s/landmark' % folder_name)
    for folder in [txt_save_dir, img_save_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    save_txt_name = 'landmark_%d.txt' % args.img_size
    save_txt_to   = os.path.join(txt_save_dir, save_txt_name)
    
    print('folder name : %s, img_size : %d' % (folder_name, args.img_size))
    print('landmark txt-file will be saved in %s' % save_txt_to)
    print('landmark img-file will be saved in %s' % img_save_dir)
    
    with open(args.anno_file, 'r') as f:
        annotations = f.readlines()
    f.close()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    f = open(save_txt_to, 'w')

    for idx, annotation in enumerate(annotations):

        annotation = annotation.strip().split(' ')

        assert len(annotation) == 6, "each line should have 6 element"  # NOTE

        im_path = os.path.join(args.data_dir, annotation[0])
        img     = cv2.imread(im_path)
        assert (img is not None)
        height, width, channel = img.shape

        pts_path = os.path.join(args.data_dir, annotation[1])
        landmark, _ = lmk_utils.anno_parser_lmks(pts_path)
        # landmark = list(map(float, annotation[5:]))
        # landmark = np.array(landmark, dtype=np.float)

        gt_box  = np.array(list(map(float, annotation[-4:])), dtype=np.int32)
        
        if (idx + 1) % 100 == 0:
            print('%d images done, landmark images: %d' % ((idx + 1), l_idx))

        x1, y1, x2, y2 = gt_box
        h, w = (y2 - y1 + 1), (x2 - x1 + 1)

        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # random shift
        for i in range(args.num_rands):

            bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box   = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (args.img_size, args.img_size), \
                                    interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_pts = np.zeros(landmark.shape, dtype=np.float)
            offset_pts[:, 0] = (landmark[:, 0] - nx1) / float(bbox_size)
            offset_pts[:, 1] = (landmark[:, 1] - ny1) / float(bbox_size)

            iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))

            if iou > 0.65:
                save_file  = os.path.join(img_save_dir, "%s.jpg" % l_idx)
                prefix_str = ' -2 %.2f %.2f %.2f %.2f ' % (offset_x1, offset_y1, offset_x2, offset_y2)
                suffix_str = ('%.2f ' * 136) % tuple(offset_pts.reshape(1, -1)[0]) + '\n'
                save_info  = save_file + prefix_str + suffix_str
                save_flag  = cv2.imwrite(save_file, resized_im)
                if not save_flag:
                    print('Attention, save %s failed, please check.' % save_file)
                f.write(save_info)
                l_idx += 1
    f.close()

def gen_config():

    parser = argparse.ArgumentParser(description=' Generate lmk file')

    parser.add_argument('--anno_file', type=str,  default=os.path.join(root_dir, '68keypoints/anno_store/300w_all.txt'))
    parser.add_argument('--data_dir',  type=str,  default=os.path.join(root_dir, '300W-Original'))
    parser.add_argument('--save_dir',  type=str,  default=os.path.join(root_dir, '68keypoints'))
    parser.add_argument('--img_size',  type=int,  default=48)    # TODO
    parser.add_argument('--num_rands', type=int,  default=45)    # TODO
    parser.add_argument('--threshold', type=float,default=0.65)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    gen_data(gen_config())
