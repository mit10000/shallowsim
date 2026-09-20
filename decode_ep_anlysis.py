#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
sys.path.insert(1, os.path.join(os.getcwd()  , '..'))
import shallowsim as sb
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import time
from tqdm import tqdm, trange


# In[2]:


args = sb.ModelArgs()
c = sb.Config()
gpu_all_decode = sb.get_gpu_info('./device/gpu_info.csv',
                                 decoding_mode=True,print_console=True) 


# In[3]:


# generate data
dfs = []
for seq_len in trange(1024,16384,32):
    c.seq_len = seq_len
    df = sb.decode_time_with_ep_list(args,gpu_all_decode,c,fp8_combine=True)
    df['index_value'] = seq_len
    df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)
    df_o.drop_duplicates(subset=['GPU','BatchSize','EP'], keep='first', inplace=True)
    dfs.append(df_o)
df = pd.concat(dfs)    
df.reset_index(inplace=True,drop=True)
df.to_csv('perf_vs_seq_len.csv')


# In[4]:


df = pd.read_csv('perf_vs_seq_len.csv')


# In[5]:


#df1 = df[df['EP'] == 144].reset_index(drop=True)
df1 = df[df['BatchSize'] == 128].reset_index(drop=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


#gpu_all_decode = sb.get_gpu_info('./device/gpu_info.csv',
#                                 device_list=['GB300-NVL72','H800','H20'],
#                                 decoding_mode=True) 

sb.draw(df1, gpu_all_decode, 
        comp_name='EP',comp_val_list=[36,72,144,320],
        val_list=['Total','TPS'],val_unit_name='Token per second',
        title='seq_len under EP strategies',savefig=True,filename='seq_len.png')


# In[7]:


sb.draw(df1, gpu_all_decode, 
        comp_name='EP',comp_val_list=[36,72,144,320],
        val_list=['SparseMLA','Delta'],val_unit_name='ms',
        title='seq_len under EP strategies')


# In[ ]:




