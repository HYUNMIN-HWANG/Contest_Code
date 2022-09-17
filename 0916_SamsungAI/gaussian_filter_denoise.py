import cv2
import pandas as pd
import os
import numpy as np
from PIL import Image
import glob

"""
# 1. test REAL SEM에 gaussian filter 적용
sim_sem_dir = "D:\\Data\\3D_Metrology\\SEM_cyclegan\\log_result\\test\\png"
sim_test_meta = pd.read_csv(sim_sem_dir+"\\test_sim_sem_meta.csv")

result_dir_test = "D:\\Data\\3D_Metrology\\submission\\test_gaussianfilter"

for idx in range(len(sim_test_meta)) :
    path = sim_test_meta.iloc[idx]
    img = Image.open(path[0]).convert('L')
    img = np.array(img)

    denoise_img = cv2.GaussianBlur(img, (0, 0), 0.5)
    denoise_img = Image.fromarray(denoise_img)
    
    denoise_img.save(os.path.join(result_dir_test, '%06d.png' % idx), cmap='gray')
"""

"""
# 2. SIM_SEM 에 gaussian filter 적용
sim_dir = 'D://Data//3D_Metrology//simulation_data/'
sim_test_meta = pd.read_csv(sim_dir+"\\simulation_meta_3.csv")
SIM_SEM_list = sim_test_meta['SEM']

result_dir_test = "D:\\Data\\3D_Metrology\\gaussian_filter\\SIM_SEM_gaussian"

import ipdb; ipdb.set_trace()
for idx in range(len(SIM_SEM_list)) :
    path = SIM_SEM_list.iloc[idx]
    img = Image.open(path).convert('L')
    img = np.array(img)

    denoise_img = cv2.GaussianBlur(img, (0, 0), 0.5)
    denoise_img = Image.fromarray(denoise_img)
    
    denoise_img.save(os.path.join(result_dir_test, '%06d.png' % idx), cmap='gray')
"""

# 생성된 depth map에 gaussian filter 적용
submission0906_depthmap = "D:\\Data\\3D_Metrology\\submission\\submission0906"
result_dir_test = "D:\\Data\\3D_Metrology\\submission\\submission0906_gaussian"

depth_files  = glob.glob(submission0906_depthmap+'/*.png')

# import ipdb; ipdb.set_trace()
for idx in range(len(depth_files)) :
    path = depth_files[idx]
    img = Image.open(path).convert('L')
    img = np.array(img)

    denoise_img = cv2.GaussianBlur(img, (0, 0), 0.5)
    denoise_img = Image.fromarray(denoise_img)
    
    denoise_img.save(os.path.join(result_dir_test, '%06d.png' % idx), cmap='gray')
