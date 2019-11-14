#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/11/11
author: relu
'''

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from .occ_utils import OcclusionEval

from IPython import embed


class XGBClassifier(OcclusionEval):
    ''' XGBClassifier for face-occlusion task '''

    def __init__(self, args):

        OcclusionEval.__init__(self, args)
        self.args = args
        self.face_feat = None
        self.params = self._set_params()
        self.xgb_classifier = None


    def _set_params(self):

        params = {'booster':'gbtree', 'objective': 'binary:logitraw', 'eval_metric': 'auc', \
                  'silent':1, 'eta': 0.2, 'gamma':0.1, 'max_depth':6, 'lambda':1e2, \
                  'seed':0, 'subsample':0.85, 'subcolumn' : 0.85, 'colsample_bytree':0.75}
        return params


    def _feat_trans(self):
        '''
        Translate the 256-bins feature into target-bins feature

        step - 1. trans the str into int or float
        step - 2. trans the 256-feat into target-feat
        '''

        refine_feat = []
        self.face_feat.iloc[:, 1:3]  = self.face_feat.iloc[:, 1:3].astype('int')
        self.face_feat.iloc[:, 3:] = self.face_feat.iloc[:, 3:].astype('float')
        self.face_feat.iloc[:, 3:] = self.face_feat.iloc[:, 3:].fillna(0)   # TODO
        self.face_feat[2] = self.face_feat[2].apply(lambda x:1 if x == -1 else 0)

        if self.args.num_bins != 256:
            unit_len  = int(256 / self.args.num_bins)
            for idx, row in self.face_feat.iterrows():

                feat = [row[0], row[1], row[2]]
                trans_feat = np.zeros((3, self.args.num_bins))
                raw_feat = np.array(row[3:]).reshape(3, -1)
                for i in range(self.args.num_bins):

                    s_idx = i * unit_len
                    e_idx = (i + 1) * unit_len
                    trans_feat[:, i] = raw_feat[:, s_idx:e_idx].mean(axis=1)
                trans_feat = trans_feat.reshape(1, -1)[0].tolist()
                feat.extend(trans_feat)
                refine_feat.append(feat)
            self.face_feat = pd.DataFrame(refine_feat)
        print('Feat-trans was finished ... shape : ', self.face_feat.shape)


    def _extract_feat(self, reproduce = False):
        ''' Extract the img_feat for each face_file in csv_file  '''

        save_dir = os.path.join(self.args.data_dir, 'csv_raw/face_feat')

        if reproduce:
            face_feat = []
            df_file   = self._prepare_csv()
            for idx, row in df_file.iterrows():
                det_flag, img_feat = self._bins_fea(row['img_path'])
                if det_flag:
                    face_info = [row['img_type'], row['anno_label']]
                    face_info.extend(img_feat.reshape(1, -1)[0].tolist())
                    face_feat.append(face_info)
                if (idx + 1) % 20 == 0:
                    print('Already processed %3d images, total %3d' % (idx+1, df_file.shape[0]))
            self.face_feat = pd.DataFrame(face_feat)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.save(os.path.join(save_dir, 'face_feat_1111.npy'), face_feat)
        else:
            face_feat = np.load(os.path.join(save_dir, 'face_feat_1111.npy'))
            self.face_feat = pd.DataFrame(face_feat)
            print('face_feat.shape : ', self.face_feat.shape)
            self._feat_trans()
        print('Extracting feature module was finished ...')


    def _split_dataset(self, split_ratio = 0.6):
        ''' Split the face-feat randomly into df_train and df_test '''

        # num_train  = int(self.face_feat.shape[0] * split_ratio)
        # df_shuffle = self.face_feat.sample(frac=1.0)
        # df_shuffle.index = range(self.face_feat.shape[0])
        # dtrain    = xgb.DMatrix(df_shuffle.iloc[:num_train, 3:], label=df_shuffle.iloc[:num_train, 2])
        # dtest, y  = xgb.DMatrix(df_shuffle.iloc[num_train:, 3:]), df_shuffle.iloc[num_train:, 2]

        df_dana = self.face_feat[self.face_feat[1] == 0]
        df_aku  = self.face_feat[self.face_feat[1] == 1]
        dtrain    = xgb.DMatrix(df_dana.iloc[:, 3:].astype(float), label=df_dana.iloc[:, 2])
        dtest, y  = xgb.DMatrix(df_aku.iloc[:, 3:].astype(float)), df_aku.iloc[:, 2]
        return dtrain, dtest, y


    def _train_xgbclassifier(self, dtrain, num_round = 40):
        ''' Train a xgb_classifier '''

        self.xgb_classifier = xgb.train(self.params, dtrain, num_round)
        print('xgb_classifier was trained ...')


    def _eval_xgbclassifier(self, dtest, gt_label):
        ''' Test the trained xgb_classifier '''

        pred_score  = self.xgb_classifier.predict(dtest)  # TODO :: fetch label out
        pred_label  = (pred_score >= self.args.xgb_thres) * 1
        res_message = metrics.classification_report(gt_label, pred_label, digits=4)
        print(res_message)


    def runner(self, num_exps = 10, reproduce = False):

        self._extract_feat(reproduce=reproduce)

        for k in range(num_exps):

            print('experiment %2d is going to start ...' % (k+1))
            dtrain, dtest, y = self._split_dataset(split_ratio=0.6)

            self._train_xgbclassifier(dtrain)
            eval_info = self._eval_xgbclassifier(dtest, y)
