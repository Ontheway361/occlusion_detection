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

root_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ'

    
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
        df_csv.drop_duplicates(subset=['source', 'input'], keep='last', inplace=True)
        df_csv = df_csv[df_csv['anno_label'] == self.args.map_key]
        self.df_csv = df_csv
        

    def _download_images(self):
        ''' Downloading the url_imgs '''
        
        badurl_file = osp.join(self.args.root_dir, 'bad_url/%s.txt' % \
                                   self.args.url_csv.split('.')[0])
        badurl_f  = open(badurl_file, 'a+')
        goodurl_count, badurl_count = 0, 0
        for idx, row in self.df_csv.iterrows():
            
            source_url, input_url = row['source'], row['input']
            source_url.rstrip('\n')
            input_url.rstrip('\n')
            
            try:
                folder_name = 'wrong_pair_%d' % goodurl_count
                save_dir    = osp.join(self.args.root_dir, 'wrong_pair/%s'% folder_name)
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_s_img  = osp.join(save_dir, 'source_%d.jpg' % goodurl_count)
                save_i_img  = osp.join(save_dir, 'input_%d.jpg' % goodurl_count) 
                
                s_img = request.urlretrieve(source_url, save_s_img)
                i_img = request.urlretrieve(input_url,  save_i_img)
            except:
                error_info = source_url + ' ' + input_url + '\n'
                badurl_f.write(error_info)
                badurl_count += 1
            else:
                goodurl_count += 1
  
        badurl_f.close()
        print('there are %d urls, good : %d, bad : %d' % \
              (self.df_csv.shape[0], goodurl_count, badurl_count))
                
        
    def runner(self):
        ''' Pipeline of Fetch_URL_Image '''
        
        self._set_header()
        self._preprocess()
        self._download_images()
    

def urlimgs_config():
    
    parser = argparse.ArgumentParser(description='Download URL images')
    
    parser.add_argument('--root_dir', type=str, default=root_dir)
    parser.add_argument('--url_csv',  type=str, default='1024_night.csv')
    parser.add_argument('--map_key',  type=str, default=1)
    parser.add_argument('--timeout',  type=int, default=10)

    args = parser.parse_args()

    return args
    

if __name__ == '__main__':
    
    url_engine = Fetch_URL_Image(urlimgs_config())
    df_occ     = url_engine.runner()