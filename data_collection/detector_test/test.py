#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:16:13 2020

@author: yueshan
"""
import cv2
import numpy as np
def detector(seg_img): 
    boxes=np.array([[]]).reshape(0,4)
    classes=np.array([])
    #set detector in BGR
    colors=[(226,117,100),(101,111,226),(117,114,116),(15,171,216)]
    img_detected=seg_img
    for obj in range(4):
        tol=0
        color=colors[obj]        
        lower=np.array((color[0]-tol,color[1]-tol,color[2]-tol),dtype=np.int32)
        upper=np.array((color[0]+tol,color[1]+tol,color[2]+tol),dtype=np.int32)
    
        mask = cv2.inRange(seg_img, lower, upper)

        result=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=result[0]
        
        for contour in contours:
            if(contour.shape[0]>3):
                x,y,w,h=cv2.boundingRect(np.vstack(contour).squeeze())
                if w>5 and h>5:
                    boxes=np.append(boxes,[[x,y,x+w,y+h]],axis=0)
                    classes=np.append(classes,obj+1)
                    #img_detected=cv2.rectangle(img_detected,(x,y),(x+w,y+h),(0,255,0),1)
    for box in boxes:
        img_detected=cv2.rectangle(img_detected,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
        
    cv2.imshow('rect',img_detected)
    return img_detected

img=cv2.imread('test9.png')

kernal=np.ones((4,4),np.uint8)
img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernal)

res=detector(img)
