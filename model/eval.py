#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:05:43 2020

@author: yueshan
"""
from map_sol import mean_average_precision
import os
import numpy as np
import cv2
    
from tqdm import trange

from model import Wrapper


def make_batches(list,BATCH_SIZE):
    for i in range(0, len(list), BATCH_SIZE):
        yield list[i:i+BATCH_SIZE]
        
def make_boxes(id, labels, scores, bboxes):
    temp = []
    for i in range(len(labels)):
        x1 = bboxes[i][0]
        y1 = bboxes[i][1]
        x2 = bboxes[i][2] - x1
        y2 = bboxes[i][3] - y1

        temp.append([id, labels[i], scores[i], x1, y1, x2, y2])
    return temp

def draw_prediction(id,img,t_boxes,t_classes,p_boxes,p_classes,p_scores):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_shape=img.shape[0]*3,img.shape[1]*3
    img=cv2.resize(img, img_shape)
    thickness=1
    box_num=0
    for box in p_boxes:
        box=box.astype(int)
        
        if p_classes[box_num]==1:# is duckie
            color=(226,117,100)
        elif p_classes[box_num]==2:#is cone
            color=(101,111,226)
        elif p_classes[box_num]==3:#is truck
            color=(117,114,116)
        elif p_classes[box_num]==4:#is bus
            color=(15,171,216)
            
        img=cv2.rectangle(img,tuple(box[0:2]*3),tuple(box[2:4]*3),color,thickness)
        img=cv2.putText(img,str(p_scores[box_num]),(box[0]*3,box[1]*3),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        box_num+=1
        
    box_num=0
    for box in t_boxes:          
        img=cv2.rectangle(img,tuple(box[0:2]*3),tuple(box[2:4]*3),(0,255,0),thickness)
        box_num+=1
    
    filename=os.path.join('prediction',str(id)+'.bmp')
    cv2.imwrite(filename,img)
    
def main():
    dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./newstyle_validation_trainset")))
    
    wrapper = Wrapper()
    
    true_boxes = []
    pred_boxes = []
    
    BATCH_SIZE = 2
    BATCH_QTTY = int(len(dataset_files) / BATCH_SIZE)
    
    batches = list(make_batches(dataset_files[:BATCH_QTTY*BATCH_SIZE],BATCH_SIZE))
    for nb_batch in trange(len(batches)):
        batch = batches[nb_batch]
    
        for nb_img, file in enumerate(batch):
            with np.load(f'./newstyle_validation_trainset/{file}') as data:
                img, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])
    
                p_boxes, p_classes, p_scores = wrapper.predict(np.array([img]))
                
                draw_prediction(nb_batch*BATCH_SIZE+nb_img,img,boxes,classes,p_boxes[0],p_classes[0],p_scores[0])
                
                for j in range(len(p_boxes)):
                    pred_boxes += make_boxes(nb_batch+nb_img, p_classes[j], p_scores[j], p_boxes[j])
                true_boxes += make_boxes(nb_batch+nb_img, classes, [1.0]*len(classes), boxes)
    
    true_boxes = np.array(true_boxes, dtype=float)
    pred_boxes = np.array(pred_boxes, dtype=float)
    # print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5))
    print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5).item())

if __name__ == "__main__":
    main()
