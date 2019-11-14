#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import socket
import argparse
import numpy as np
import pandas as pd
from os import path as osp
import urllib.request as request

from IPython import embed

# root_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ' # gpu3
root_dir = '/Users/relu/data/deep_learning/face_landmark/imgs_occ/'  # local


class Fetch_URL_Image(object):

    def __init__(self, args):

        self.args   = args
        self.df_csv = None

    def _set_header(self):
        ''' Set the header for url_request '''

        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'
        headers = ('User-Agent', user_agent)
        opener = request.build_opener()
        opener.addheaders = [headers]
        request.install_opener(opener)
        socket.setdefaulttimeout(self.args.timeout)  # wait seconds


    def _preprocess(self):
        ''' Preprocess the raw url_csv file '''

        df_csv = pd.read_csv(osp.join(self.args.root_dir, 'csv_raw', self.args.url_csv))
        df_csv.drop_duplicates(subset=['img_url'], keep='last', inplace=True)
        #df_csv = df_csv[df_csv['anno_label'] == self.args.map_key]
        self.df_csv = df_csv


    def _download_images(self):
        ''' Downloading the url_imgs '''

#         badurl_file = osp.join(self.args.root_dir, 'bad_url/%s.txt' % \
#                                    self.args.url_csv.split('.')[0])
#         badurl_f  = open(badurl_file, 'a+')
        folder_name = self.args.note_csv.split('_')[-1].split('.')[0]
        save_dir = osp.join(self.args.root_dir, 'faces_%s' % folder_name)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        goodurl_count =  0
        note_info = []
        for idx, row in self.df_csv.iterrows():

            img_url = row['img_url'].rstrip('\n')
            save_img  = osp.join(save_dir, '%d.jpg' % goodurl_count)

            try:
                s_img = request.urlretrieve(img_url, save_img)
            except:
                print('Downloading %s failed ...' % img_url)
            else:
                goodurl_count += 1
                note_info.append([save_img, row['img_type'], row['anno_label']])
            print('%d url was processed ...' % idx)
        df_note = pd.DataFrame(note_info, columns=['img_path', 'img_type', 'anno_label'])
        df_note.to_csv(osp.join(self.args.root_dir, 'csv_raw', self.args.note_csv), index=None)
        print('there are %d urls, good : %d' % (self.df_csv.shape[0], goodurl_count))


    def runner(self):
        ''' Pipeline of Fetch_URL_Image '''

        self._set_header()
        self._preprocess()
        self._download_images()


def urlimgs_config():

    parser = argparse.ArgumentParser(description='Download URL images')

    parser.add_argument('--root_dir', type=str, default=root_dir)
    parser.add_argument('--url_csv',  type=str, default='self_anno_occ_20191106.csv')
    parser.add_argument('--note_csv', type=str, default='note_1106.csv')
    parser.add_argument('--map_key',  type=str, default=1)
    parser.add_argument('--timeout',  type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    url_engine = Fetch_URL_Image(urlimgs_config())
    df_occ     = url_engine.runner()
