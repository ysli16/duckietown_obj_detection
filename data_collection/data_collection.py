import numpy as np
import cv2
import os
from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./new_dataset"):
        np.savez(f"./new_dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1
        print("saved image ",npz_index)

def clean_segmented_image(img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    hls_img=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

    #change background color to black
#    for i in range(hls_img.shape[0]):
#        for j in range(hls_img.shape[1]):
#            if hls_img[i,j,0]==150:
#                hls_img[i,j]=[0,0,0]
#    img=cv2.cvtColor(hls_img,cv2.COLOR_HLS2BGR)
#    kernal=np.ones((2,2),np.uint8)
#    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernal)
#    hls_img=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

    boxes,classes=detector(img,hls_img)
    return boxes, classes

tol=10
#detect duckie in hsl mode
lower_duckie=np.array((116,255*0.3,255*0.3),dtype=np.int32)
upper_duckie=np.array((116,255*0.9,255*0.9),dtype=np.int32)
#detect cone in hsl mode
lower_cone=np.array((2,255*0.3,255*0.3),dtype=np.int32)
upper_cone=np.array((2,255*0.9,255*0.9),dtype=np.int32)
#detect truck in rgb mode
lower_truck=np.array((117-tol,114-tol,116-tol),dtype=np.int32)
upper_truck=np.array((117+tol,114+tol,116+tol),dtype=np.int32)
#deteect bus in hsl mode
lower_bus=np.array((23,255*0.3,255*0.3),dtype=np.int32)
upper_bus=np.array((23,255*0.9,255*0.9),dtype=np.int32)

lower=[lower_duckie,lower_cone,lower_truck,lower_bus]
upper=[upper_duckie,upper_cone,upper_truck,upper_bus]

def detector(img,hls_img): 
    boxes=np.array([[]]).reshape(0,4).astype(int)
    classes=np.array([]).astype(int)
    mask_img=np.zeros((224,224))
    #set detector in BGR
    #colors=[(226,117,100),(101,111,226),(117,114,116),(15,171,216)]
    #img_detected=seg_img
    for obj in range(4):
        lb=lower[obj]
        ub=upper[obj]
        if obj==2:
            mask = cv2.inRange(img, lb, ub)
        else:
            mask = cv2.inRange(hls_img, lb, ub)
    
        mask_img=mask_img+mask
        
        result=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=result[0]
        
        for contour in contours:
            if(contour.shape[0]>3):
                x,y,w,h=cv2.boundingRect(np.vstack(contour).squeeze())
                if obj!=2: #if object isn't truck, only discard small squres and very thin rectangles
                    if not (w<5 and h<5 or w<3 or h<3):
                        boxes=np.append(boxes,[[x,y,x+w,y+h]],axis=0)
                        classes=np.append(classes,obj+1)
                else:#if object is truck, it's easy to be confused with white lines, so thin rectangles are also discarded
                    if w>5 and h>5:
                        boxes=np.append(boxes,[[x,y,x+w,y+h]],axis=0)
                        classes=np.append(classes,obj+1)
#    for box in boxes:
#        img_detected=cv2.rectangle(img_detected,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
#        
#    cv2.imshow('rect',img_detected)
#    display_seg_mask(seg_img,mask_img)
    return boxes,classes

def save_seg(seg_img,idx):
    #cur_dir=os.getcwd()
    filename=os.path.join('seg_img',str(idx)+'.bmp')
    cv2.imwrite(filename,seg_img)
    
    
seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while npz_index<3000:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        
        if nb_of_steps%3==0:
            obs=cv2.resize(obs,(224,224))
            segmented_obs=cv2.cvtColor(segmented_obs,cv2.COLOR_RGB2BGR)
            segmented_obs=cv2.resize(segmented_obs,(224,224))
            
            boxes, classes = clean_segmented_image(segmented_obs)
            if boxes.shape[0]>0:
                print(boxes)
                save_seg(segmented_obs,npz_index)
                save_npz(obs, boxes, classes)
        
        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
