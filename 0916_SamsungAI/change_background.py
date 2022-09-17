import cv2
import pandas as pd
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pylab as plt

# submission0906_depthmap = "D:\\Data\\3D_Metrology\\submission\\submission0906"
# result_dir_test = "D:\\Data\\3D_Metrology\\submission\\submission0912_background"

# SIM_log_result_bc16 = "D:\\Data\\3D_Metrology\\SEM_cyclegan\\SIM_log_result_bc16\\test\\png"
# result_dir_test = "D:\\Data\\3D_Metrology\\submission\\submission0913_SIM_log_result_bc16"

# sem_depth_cyclegan = "D:\Data\\3D_Metrology\\SEM_cyclegan\\log_result_sem_depth\\test\\png"
# result_dir_test = "D:\\Data\\3D_Metrology\\submission\\submission0913_sem_depth_cyclegan"

submission0906_depthmap = "D:\\Data\\3D_Metrology\\submission\\submission0906"
result_dir_test = "D:\\Data\\3D_Metrology\\submission\\submission0914_0906depthmap_resnet50"

depth_files  = glob.glob(submission0906_depthmap+'/*.png')

test_dir = "D:\\Data\\3D_Metrology\\test\\SEM"
# sem_meta = pd.read_csv(test_dir+"\\test_sem_meta_class.csv")
sem_meta = pd.read_csv(test_dir+"\\test_sem_meta_class_resnet50.csv")

for idx in range(len(depth_files)) :
    path = depth_files[idx]
    img = cv2.imread(path, 0)

    # th, hole = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU ) 
    th, hole = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU ) 
    hole = cv2.erode(hole, None)

    background = 255-hole

    class_num = sem_meta['Class'][idx]

    if class_num == 0:
        background_pixel = 140
    elif class_num == 1:
        background_pixel = 150
    elif class_num == 2:    
        background_pixel = 160
    else : 
        background_pixel = 170

    change_img = cv2.copyTo(img, background)
    change_background = np.where(change_img==0, background_pixel, change_img)

    cv2.imwrite(os.path.join(result_dir_test, '%06d.png' % idx), change_background)
    