import matplotlib.pyplot as plt
import pandas as pd
import glob

data_dir = 'D://Data//3D_Metrology//simulation_data/'
depth_dir = data_dir + '/Depth//Case_'
sem_dir = data_dir + '/SEM//Case_'
case_list = ['1', '2', '3', '4']
category_list = ['//80', '//81', '//82', '//83', '//84']


f = open(data_dir+"simulation_meta.csv", "w")

meta_pd = pd.DataFrame(index=range(0), columns={'Depth','SEM'})

# depth map path
for case in case_list :
    temp_path = depth_dir + case
    print(temp_path)
    for category in category_list :
        temp2_path = temp_path + category 
        print(temp2_path)
        depth_files  = glob.glob(temp2_path+'/*.png')
        # print(depth_files)
        print(len(depth_files))
        temp_df = pd.DataFrame(depth_files, columns=['Depth'])
        meta_pd = pd.concat((meta_pd, temp_df), sort=False,ignore_index=True)
meta_pd = pd.concat((meta_pd, meta_pd), sort=False, ignore_index=True)
meta_pd = meta_pd.sort_values(by='Depth', ignore_index=True)

# sem path
# sem_files = []
# for case in case_list :
#     temp_path = sem_dir + case
#     print(temp_path)
#     for category in category_list :
#         temp2_path = temp_path + category 
#         print(temp2_path)
#         sem_files  += glob.glob(temp2_path+'/*.png')

#     for idx in range(len(meta_pd)) :
#         sem_name = sem_files[2*idx : 2*idx+2]
#         meta_pd['SEM'][idx] = sem_name

sem_files = []
for case in case_list :
    temp_path = sem_dir + case
    print(temp_path)
    for category in category_list :
        temp2_path = temp_path + category 
        print(temp2_path)
        sem_files  += glob.glob(temp2_path+'/*.png')
temp_df = pd.DataFrame(sem_files, columns=['SEM'])
temp_df = temp_df.sort_values(by='SEM')
meta_pd['SEM'] = temp_df



print(meta_pd.head())
# meta_pd.to_csv(data_dir+"simulation_meta_2.csv", index=False)
meta_pd.to_csv(data_dir+"simulation_meta_3.csv", index=False)

