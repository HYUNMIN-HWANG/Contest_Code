import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

depth_dir = 'D:\\Data\\3D_Metrology\\simulation_data\\Depth'
depth_dir = depth_dir + '//Case_'
case_list = ['1', '2', '3', '4']
category_list = ['//80', '//81', '//82', '//83', '//84']

train_dir = 'D://Data//3D_Metrology//train'
real_sem_dir = train_dir + '/SEM//Depth_'
category_list_real = ['110', '120', '130', '140']


meta_pd = pd.DataFrame(index=range(0), columns={'Depth','SEM'})

# import ipdb; ipdb.set_trace()
# simulator Depth map
for case in case_list :
    temp_path = depth_dir + case
    print(temp_path)
    for category in category_list :
        temp2_path = temp_path + category 
        print(temp2_path)
        sem_files  = glob.glob(temp2_path+'/*.png')
        # print(len(sem_files))
        temp_df = pd.DataFrame(sem_files, columns=['Depth'])
        meta_pd = pd.concat((meta_pd, temp_df), sort=False, ignore_index=True)

# REAL SEM path
sem_files = []
for case in category_list_real :
    temp_path = real_sem_dir + case
    print(temp_path)
    folder_len = len(next(os.walk(temp_path))[1])
    for number in range(folder_len) :
        temp2_path = os.path.join(temp_path,'site_00%03d' % (number))
        print(temp2_path)
        sem_files  += glob.glob(temp2_path+'/*.png')
# import ipdb; ipdb.set_trace()
temp_df = pd.DataFrame(sem_files, columns=['SEM'])
meta_pd['SEM'] = temp_df

print(meta_pd.head())
meta_pd.to_csv(train_dir+"//real_sem_depth_meta.csv", index=False)
