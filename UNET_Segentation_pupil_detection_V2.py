# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:00:51 2024

@author: Vikram
"""
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import  tensorflow as tf
# import timeit

# import keras
# from keras.models import Model
import os
from tqdm import tqdm
# import imageio
from skimage.transform import resize
import csv
import timeit
IMG_WIDTH =128
IMG_HEIGHT=128
IMG_CHANNELS=3
BATCH_SIZE=1000 #<======change this according to your Computer's RAM capacity
# Inputs to the script===============================================
file_path = filedialog.askopenfilename( title='Open the eye video',filetypes=[("Video files","*")]) 
# model = tf.keras.models.load_model("marmo_pupil_detection_UNet_Selma_021924.h5")
model = tf.keras.models.load_model("marmo_pupil_detection_UNet_Gouda_031524.h5")

# ===============================================

# file_path=r'C:/Users/Vikram/Desktop/temp/eye_rec_1_frames.avi'
video = cv2.VideoCapture(file_path)
frame_rate = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frame count:"+ str(total_frames))
frame_width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
video.set(cv2.CAP_PROP_POS_FRAMES, 1)

filename=file_path.split('/')[-1].split('.', 1)[0] +('_UNET_Result.avi')
output_file_path =os.path.join(('/'.join(file_path.split('/')[:-1]) +'/'),filename)
   
out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'MJPG'), 90, (IMG_WIDTH,IMG_HEIGHT)) 
Y_predict = np.zeros((BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
preds_test=np.zeros((BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,1),dtype=bool)
channel = np.zeros([IMG_HEIGHT,IMG_WIDTH,3], dtype=np.uint8)  
start_time=timeit.default_timer()
Current_Frame=0
fig, (ax1, ax2) = plt.subplots(1, 2)
while (Current_Frame<total_frames): 
    # Avoid any out of index errors
    if (Current_Frame+BATCH_SIZE<total_frames):
        Range_stop=Current_Frame+BATCH_SIZE
    else:
        Range_stop=total_frames
    # Read the frames and predict the results using ANN model as a batch    
    for frame_id in tqdm(range(Current_Frame,Range_stop)): 
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        # frame = video.read()
        #if the input frmame needs to be resized
        # img =resize(frame,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
        # Y_predict[frame_id-Current_Frame]=img
        #if not
        Y_predict[frame_id-Current_Frame]=frame
    preds_test=model.predict(Y_predict, verbose=0)
    Current_Frame=Range_stop+1   
    # print("elapsed time is:",timeit.default_timer()-start_time)
    
    # start_time_video_write=timeit.default_timer()
    for i in range(len(preds_test)):
        seg_frm_remapped=np.squeeze(preds_test[i])*254 ;
        img = np.array(seg_frm_remapped, dtype=np.uint8)
        channel[:,:,0] = img;
        channel[:,:,1] = img;
        channel[:,:,2] = img;
        out.write(channel)
    # plot the image
    ax1.imshow(frame)
    ax1.set_title('Raw Image')
    ax2.imshow(channel)
    ax2.set_title('UNET Result')
    fig.suptitle(('Progress:'+str(round((Range_stop/total_frames)*100,2))+'%'))
    fig.canvas.draw()
    fig.canvas.flush_events()
    if (Range_stop==total_frames):
        break
out.release()
video.release()
# print("elapsed time for video_write is:",timeit.default_timer()-start_time_video_write)    
print("elapsed time is:",timeit.default_timer()-start_time)






