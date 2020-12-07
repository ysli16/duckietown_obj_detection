#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:10:54 2020

@author: yueshan
"""
import os
import numpy as np
import cv2

#imgname='14.npz'

cur_dir=os.getcwd() #/home/yueshan/Desktop/AMoD/RH8/obj_detection/object-detection-ex-template/data_collection

#from data
start_idx=0
end_idx=114

for idx in range(end_idx):
    #eval_file=os.path.join(os.path.dirname(cur_dir), 'eval','dataset',str(idx)+'.npz')#/home/yueshan/Desktop/AMoD/RH8/obj_detection/object-detection-ex-template/eval/*.npz
    coll_file=os.path.join(cur_dir,'new_dataset',str(idx)+'.npz')
    #data=np.load(eval_file)
    data=np.load(coll_file)
    
    img = data[f"arr_{0}"]
    boxes = data[f"arr_{1}"]
    classes = data[f"arr_{2}"]
    
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_shape=img.shape[0]*3,img.shape[1]*3
    img=cv2.resize(img, img_shape)
    box_num=0
    for box in boxes:
        thickness=2
        if classes[box_num]==1:# is duckie
            color=(226,117,100)
        elif classes[box_num]==2:#is cone
            color=(101,111,226)
        elif classes[box_num]==3:#is truck
            color=(117,114,116)
        elif classes[box_num]==4:#is bus
            color=(15,171,216)
            
        img=cv2.rectangle(img,tuple(box[0:2]*3),tuple(box[2:4]*3),color,thickness)
        box_num+=1
    
    filename=os.path.join('image',str(idx)+'.bmp')
    cv2.imwrite(filename,img)
    #cv2.imshow('data',img)
    data.close()
    
