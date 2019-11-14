#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/10/28
author: relu
'''

import os
import cv2
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import mtcnn.core.vision as vis_utils
from mtcnn.core.detect import MtcnnDetector


from IPython import embed


class HistOccBlock(object):


    def __init__(self, args):

        self.args      = args
        self.imgs_list = None
        self.model     = MtcnnDetector(self.args)
        self.pdf_list  = None


    def _prepare_folder(self):
        ''' Prepare the face-model, imgs_list '''

        # imgs_list
        folder_path = os.path.join(self.args.data_dir, self.args.folder)
        imgs_list   = []
        for img_name in os.listdir(folder_path):

            # idx = folder.split('_')[-1]
            # img_name = '%s_%d.jpg' % (self.args.img_type, int(idx))
            imgs_list.append(os.path.join(folder_path, img_name))
        self.imgs_list = imgs_list
        print('there are %d imgs in %s' % (len(imgs_list), self.args.folder))
        return imgs_list


    def _prepare_csv(self):
        ''' Prepare the face-model, imgs_list '''

        csv_file = os.path.join(self.args.data_dir, 'csv_raw', self.args.csv_file)
        df_csv = pd.read_csv(csv_file)
        print('csv_file.shape : ', df_csv.shape)
        df_test  = None

        if self.args.check_mode == 'pos':
            check_mode = -1
        elif self.args.check_mode == 'neg':
            check_mode = 0
        else:
            check_mode = 1
            print('attention, evaluate-mode was started ...')

        if check_mode < 1:
            df_csv = df_csv[df_csv['anno_label'] == check_mode]
            imgs_list = []
            for idx, row in df_csv.iterrows():

                img_name = '/'.join(row['img_path'].split('/')[-2:])
                img_path = os.path.join(self.args.data_dir, img_name)
                imgs_list.append(img_path)
            self.imgs_list = imgs_list
        else:
            df_test = []
            for idx, row in df_csv.iterrows():

                img_name = '/'.join(row['img_path'].split('/')[-2:])
                img_path = os.path.join(self.args.data_dir, img_name)
                df_test.append([img_path, row['img_type'], row['anno_label']])
            df_test = pd.DataFrame(df_test, columns=df_csv.columns)
            print('after filtering, df_test.shape : ', df_test.shape)
            print('imgs_list was prepared ...')
        return df_test


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


    def _fetch_block(self, img, bbox, landmark, lmk_flag = True):
        ''' Crop the chin_block of the detected face '''

        landmark = landmark.reshape(-1,2)

        if lmk_flag:
            left_down  = landmark[6]
            nose_point = landmark[33] # point-34 | nose
            right_down = landmark[10]
            chin_point = landmark[8]  # point-9  | chin

            x1, y1 = int(left_down[0]), int(nose_point[1])
            x2, y2 = int(right_down[0]), int(chin_point[1])
        else:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

        crop_block = img[y1:y2 + 1, x1:x2 + 1, :]

        return crop_block


    def _statistics(self, block):
        ''' Statistics the block-pixels info '''

        height, width, _ = block.shape
        num_pixels = height * width
        bgr_prob   = np.zeros((3, 256), dtype=np.int)

        for y in range(height):
            for x in range(width):

                pixel = block[y,x]
                bgr_prob[0, pixel[0]] += 1
                bgr_prob[1, pixel[1]] += 1
                bgr_prob[2, pixel[2]] += 1
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

        pdf_list = []
        for img_path in self.imgs_list:

            try:
                img = cv2.imread(img_path)
                bboxes, landmarks = self.model.detect_face(img, verbose=False)
                # save_name = 'result/r_%s' % img_path.split('/')[-1]
                # vis_utils.visual_face(img, bboxes, landmarks, save_name)
                print((img_path, bboxes.shape))
            except Exception as e:
                print(e)
            else:
                if bboxes.shape[0] == 0:
                    print('No face detected in %s' % img_path)
                    continue
                else:
                    bbox, landmark = self._fliter_doc_bbox(bboxes, landmarks)
                block = self._fetch_block(img, bbox, landmark)
                bgr_prob = self._statistics(block)
                pdf_list.append(bgr_prob.reshape(1, -1)[0])

        self.pdf_list = pdf_list
        date_stamp = self.args.csv_file.split('_')[-1].split('.')[0]
        save_name  = 'pdf/npy_data/%s_details_%s.npy' % (self.args.check_mode, date_stamp)
        print('npy_data was savd in %s' % save_name)
        np.save(save_name, pdf_list)
        print('hist-module was finished ...')


    def runner(self, vis = False):
        ''' Pipeline of HistOccBlock '''

        self._prepare_csv()
        self._hist_go()


class OcclusionEval(HistOccBlock):

    ''' Use statistics-info to evaluate wether there is a occlusion '''

    def __init__(self, args):

        HistOccBlock.__init__(self, args)
        self.args = args
        self.stif = dict()


    def _stif_bins(self, stif):
        ''' Statistic the info according to the num_bins '''

        stif = stif.fillna(0.0)
        if self.args.num_bins != 256:
            unit_len = int(256/self.args.num_bins)
            res_stif = np.zeros((2, self.args.num_bins * 3))
            for i in range(res_stif.shape[1]):

                s_idx = i * unit_len
                e_idx = (i+1) * unit_len
                unit_data = stif.iloc[:,s_idx:e_idx].to_numpy().reshape(-1,1)
                res_stif[0, i] = unit_data.mean(axis=0)[0]
                res_stif[1, i] = unit_data.std(axis=0)[0]
            new_stif = res_stif.reshape(-1, self.args.num_bins)
        else:
            data_mean = stif.mean(axis=0).to_list()
            data_std  = stif.std(axis=0).to_list()
            data_mean.extend(data_std)
            new_stif = np.array(data_mean).reshape(6,-1)

        return new_stif


    def _stif_load(self, vis=False):
        '''
        Load the statistics-info

        step - 1. load the npy_data
        step - 2. split the data according to the num_bins
        step - 3. form the statistics-info(stif)
        '''

        pos_stif = np.load('pdf/npy_data/pos_details_%s.npy' % self.args.anno_date)
        neg_stif = np.load('pdf/npy_data/neg_details_%s.npy' % self.args.anno_date)

        self.stif['pos_stif'] = self._stif_bins(pd.DataFrame(pos_stif))
        self.stif['neg_stif'] = self._stif_bins(pd.DataFrame(neg_stif))
        if vis:
            vis_utils.visual_prob(self.stif['pos_stif'][:3,:], 'pdf/pos_mean.jpg')
            vis_utils.visual_prob(self.stif['pos_stif'][3:6,:], 'pdf/pos_std.jpg')
            vis_utils.visual_prob(self.stif['neg_stif'][:3,:], 'pdf/neg_mean.jpg')
            vis_utils.visual_prob(self.stif['neg_stif'][3:6,:], 'pdf/neg_std.jpg')
            print('just for vis the hist info ...')
            return


    def _bins_fea(self, img_path):
        ''' Extract the bins-fea for test image '''

        stif      = np.zeros((3, self.args.num_bins))
        run_flag  = False
        unit_len  = int(256 / self.args.num_bins)
        try:
            img = cv2.imread(img_path)
            bboxes, landmarks = self.model.detect_face(img, verbose=False)
        except Exception as e:
            print(e)
        else:
            if bboxes.shape[0] == 0:
                print('No face detected in %s' % img_path)
            else:
                bbox, landmark = self._fliter_doc_bbox(bboxes, landmarks)
                block = self._fetch_block(img, bbox, landmark)
                bgr_prob = self._statistics(block)
                run_flag = True

                for i in range(self.args.num_bins):

                    s_idx = i * unit_len
                    e_idx = (i + 1) * unit_len
                    stif[:, i] = bgr_prob[:, s_idx:e_idx].mean(axis=1)

        res_cache = [run_flag, stif]
        return res_cache


    def _cal_likelihood(self, img_path):
        ''' Calculate the likelihood for test_img '''

        det_flag, img_feat = self._bins_fea(img_path)
        pos_value, neg_value = 0, 0
        if det_flag:
            pos_diff = (self.stif['pos_stif'][:3, :] - img_feat) / (np.sqrt(2) * self.stif['pos_stif'][3:6, :])
            neg_diff = (self.stif['neg_stif'][:3, :] - img_feat) / (np.sqrt(2) * self.stif['neg_stif'][3:6, :])
            # mean
            pos_value += (np.power(pos_diff, 2).sum(axis=1) * np.array(self.args.c_weights)).sum()
            neg_value += (np.power(neg_diff, 2).sum(axis=1) * np.array(self.args.c_weights)).sum()

            # # # std
            # pos_value += np.log(self.stif['pos_stif'][3:6, :]).sum()
            # neg_value += np.log(self.stif['neg_stif'][3:6, :]).sum()
        likelihood = [det_flag, -pos_value, -neg_value]  # log-likelihood
        return likelihood


    def _cal_dist(self, img_path):
        ''' Calculate the dist of mean between ave_var and test_var '''

        det_flag, img_feat = self._bins_fea(img_path)
        pos_dist, neg_dist = 0, 0
        if det_flag:
            pos_diff = self.stif['pos_stif'][:3, :] - img_feat
            neg_diff = self.stif['neg_stif'][:3, :] - img_feat

            pos_dist = np.linalg.norm(pos_diff, ord=self.args.order, axis=1)
            neg_dist = np.linalg.norm(neg_diff, ord=self.args.order, axis=1)

            pos_dist = (pos_dist * np.array(self.args.c_weights)).sum()
            neg_dist = (neg_dist * np.array(self.args.c_weights)).sum()
        res_dist = [det_flag, pos_dist, neg_dist]
        return res_dist


    def _eval_occ(self):
        '''
        Evaluate the occlusion detection algorithm

        step - 1. construct the test_dataset
        step - 2. run occ_detection algorithm
        step - 3. pretty show the result
        '''

        # df_test = self._prepare_csv()
        imgs_list = self._prepare_folder()
        # gt_label, pred_label = [], []
        # for idx, row in df_test.iterrows():
        idx = 0
        for img_path in imgs_list:

            # run_flag, pos_ll, neg_ll = self._cal_likelihood(row['img_path'])
            run_flag, pos_ll, neg_ll = self._cal_likelihood(img_path)
            # run_flag, pos_ll, neg_ll = self._cal_dist(row['img_path'])
            if run_flag:
                rre_ll = (pos_ll - neg_ll) / abs(neg_ll)  # TODO
                pred_y = -1 if rre_ll > self.args.rre_thres else 0  # TODO  ： -0.5

            if pred_y == 0 or (not run_flag):
                img_name = img_path.split('/')[-1]
                target_path = os.path.join(self.args.data_dir, 'check_out', img_name)
                shutil.move(img_path, target_path)
            idx += 1
            if idx % 100 == 0:
                print('already processed %4d|%4d ...' % (idx, len(imgs_list)))
            
        #         gt_label.append(row['anno_label'])
        #         pred_label.append(pred_y)
        #         show_str = 'img_info : %s, gt_label: %2d, | pred : %2d, (pos_ll-neg_ll)/|neg_ll| : %.4f' % \
        #                    (row['img_path'].split('/')[-1], row['anno_label'], pred_y, rre_ll)
        #         print(show_str)
        # print(metrics.classification_report(gt_label, pred_label, digits=4))
        # print(metrics.confusion_matrix(gt_label, pred_label))



    def runner(self, vis=False):

        self._stif_load(vis)

        self._eval_occ()
