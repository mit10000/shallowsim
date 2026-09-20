#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
sys.path.insert(1, os.path.join(os.getcwd()  , '..'))


# In[2]:


import shallowsim as sb
import pandas as pd
import math


# In[3]:


args = sb.ModelArgs()
gpu_blackwell = sb.get_gpu_info('./device/gpu_info.csv',print_console=True ) 


# In[4]:


seq_len = 4383
kv_cache_rate = 0.563
decode_len = 1210
bs_list =[ 16, 32, 64, 128, 256, 512]
eplist = [ 8 , 16, 36, 72, 144, 320]


# In[5]:
#only 计算 H800
gpu_H800 = {'H800' : gpu_blackwell['H800']}

#detail,summary = sb.prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=4, dp=8)

detail,summary = sb.prefill_time(args,gpu_H800,seq_len, kv_cache_rate, tp=4, dp=8)

# In[6]:


print(detail)


# In[7]:


print(summary)


# In[8]:


tp=4
_ , ttft_sum = sb.prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=tp, dp=8, print_console=False)
print(ttft_sum.apply(lambda x: seq_len/tp * (1000/ x)).loc['Sum'].to_markdown(floatfmt=".1f"))


# In[ ]:




