#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from IPython import embed



def visual_prob(prob_array, pdf_dir, num_bins = 64):
    ''' Visual the curve of pixel_prob '''
    
    for idx, prob in enumerate(prob_array):
        
        plt.hist(prob, bins=num_bins, density=True)
        pdf_name = os.path.join(pdf_dir, 'pdf_%d.jpg' % (idx+1))
        plt.savefig(pdf_name)
        


def visual_face(img, bboxes, landmarks, save_name, re_scale = 1, fontScale = 32):
    ''' Visualize detection results with on axis '''
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', fontScale)
    except:
        #font_path = os.path.join(os.environ['HOME'], '.fonts', 'freefont', 'FreeMono.ttf')
        font_path = os.path.join('/usr/share/fonts/truetype/liberation', 'LiberationMono-BoldItalic.ttf') # gpu_server
        font = ImageFont.truetype(font_path, fontScale)
    
    for i in range(bboxes.shape[0]):
        
        bbox = bboxes[i, :]
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='yellow')
        top_y = max(0, bbox[1]-10)
        draw.text((bbox[0], top_y), '%.4f' % bbox[4], fill='blue')
    
    if landmarks is not None:
        
        for i in range(landmarks.shape[0]):
            
            pts  = landmarks[i, :].reshape((68, 2))
            for idx in range(68):
                
                point = (pts[idx,0]-1, pts[idx,1]-1, pts[idx,0]+1, pts[idx,1]+1)
                draw.ellipse(point, fill=None, outline='green')
                
    width, height = pil_img.size
    new_size = (int(width*re_scale), int(height*re_scale))
    pil_img = pil_img.resize(new_size)
    pil_img.save(save_name)

