import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


"""sim_dir = 'D://Data//3D_Metrology//test//SEM'

meta_pd = pd.DataFrame(index=range(0), columns={'REAL'})

# Test SEM path
sem_files  = glob.glob(sim_dir+'/*.png')
meta_pd = pd.DataFrame(sem_files, columns=['REAL'])

print(meta_pd.head())
meta_pd.to_csv(sim_dir+"//test_sem_meta.csv", index=False)"""


sim_sem_dir = "D:\\Data\\3D_Metrology\\SEM_cyclegan\\log_result_bc16\\test\\png"
meta_pd = pd.DataFrame(index=range(0), columns={'SEM'})

sim_sem_files  = glob.glob(sim_sem_dir+'/*.png')
meta_pd = pd.DataFrame(sim_sem_files, columns=['SEM'])

print(meta_pd.head())
meta_pd.to_csv(sim_sem_dir+"//test_sim_sem_meta.csv", index=False)