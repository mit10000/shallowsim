#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
import warnings
sys.path.insert(1, os.path.join(os.getcwd()  , '..'))
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# In[2]:


import shallowsim as sb
import pandas as pd
import math
from pdb import set_trace as bp

# In[3]:


args = sb.ModelArgs()
c = sb.Config()
config=sb.Config
#c.bs_list=[16,32,64,128,256,512,1024,2048]
#c.bs_list=[1,2,4,8,16,32,64,128,256,512]#[1]
c.bs_list=[1,8]

#设置SLO 最低限制
SLOs = None
#SLOs = {"TPS":30,}
#是否使用2.7T DeepSeek模型
User2p7T = False#True#False#True
if User2p7T:
      args.n_layers = 235#888#10T模型


gpu_blackwell_decode,GPUs = sb.get_gpu_info('./device/gpu_info.csv', 
                                    decoding_mode=True, print_console=True) 
#bp()
#gpu_blackwell_decode['H20'].pcie_bw = 25 #change bw to 1.6Tbps..

#bp()
# In[6]:
#gpu_H800_decode = {'H800' : gpu_blackwell_decode['H800']}
gpu_B200_decode = {'B200' : gpu_blackwell_decode['DGX-B200']}
#HBF，latency完全隐藏 0.0001ms
#gpu_B200_decode = {'B200-HBF-lowLatency' : gpu_blackwell_decode['B200-HBF-lowLatency'],'B200' : gpu_blackwell_decode['DGX-B200']}
#gpu_B200_decode = {'B200-HBF-highLatency' : gpu_blackwell_decode['B200-HBF-highLatency'],'B200' : gpu_blackwell_decode['DGX-B200']}
#gpu_B200_decode = {'B200-HBF-lowLatency' : gpu_blackwell_decode['B200-HBF-lowLatency'],}
#HBF，latency不能隐藏 0.1ms
#gpu_B200_decode = {'B200-HBF-highLatency' : gpu_blackwell_decode['B200-HBF-highLatency'],'B200' : gpu_blackwell_decode['DGX-B200']}

#生成比较表格用
gpu_map = {"DGX-B200": "B200",  "B200-HBF-lowLatency": "B200-HBF"}

#dfs = sb.decode_time_with_ep_list(args,gpu_blackwell_decode,c,print_console=False,fp8_combine=True,tps_limit=0)
#是否考虑HBM/HBF memory 读latency
mem_read_latency = True#False
dfs = sb.decode_time_with_ep_list(args,gpu_B200_decode,c,print_console=False,fp8_combine=True,tps_limit=0,mem_read_latency=mem_read_latency)
#print(dfs)
#bp()
#准备对比表格的数据，挑选df里面的行
#best_row = dfs.loc[(dfs["EP"] == 16) & (dfs["GPU"]=='DGX-B200')].loc[lambda x: x["Total"].idxmax()]

# # 推理性能天梯榜

# In[7]:
"""
#这里通过groupby，删除了很多row
对于每种 (GPU + BatchSize) 的组合，
仅仅保存组合里面 Total 最大的那个行
然后把所有组合的最优结果按 Total 从高到低排列。
"""
#bp()
# 根据SLO lower bound 删除不符合条件的
if SLOs:
      for key,val in SLOs.items():
             dfs=dfs[dfs[key]>val]
# 每个GPU/Batchsize/EP里面，对应TP=1，4，8，仅仅保存里面Total最大的行
dfs_o = dfs
#dfs_o = dfs.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)
#dfs_o = dfs.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)



dfs_o.style.bar(subset=['TPS','Total'],color='#6495ED')\
      .applymap(sb.gpu_category_color,props=sb.gpu_category_idx(gpu_blackwell_decode),subset=['GPU'])\
      .applymap(sb.color_positive_red, subset=['Delta'])\
      .background_gradient(subset=['Comm_Impact'],cmap=sb.cm)\
      .format(precision=3)\
      .to_html()
#bp()
print(dfs_o)
#存为html才能显示
# 1. Generate the Styler object，推理性能信息，
styler = (
    dfs_o.style.bar(subset=['TPS', 'Total'], color='#6495ED')
    .applymap(
        sb.gpu_category_color,
        props=sb.gpu_category_idx(gpu_blackwell_decode),
        subset=['GPU'],
    )
    .applymap(sb.color_positive_red, subset=['Delta'])
    .background_gradient(subset=['Comm_Impact'], cmap=sb.cm)
    .format(precision=3)
)
"""
# 2. Render to HTML string
html_content = styler.to_html()

# 3. Save to an .html file
with open("output_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
 
#dfs_o.to_html("data_bar.html")
"""
#拼接G PU 信息，和 前面性能分析数据，到一个html里面
styler2 = (GPUs.style
    #.background_gradient(cmap='Blues', subset=[''])  # 色阶
    .set_caption("GPU parameters")
)
# 拼接两个styled html
html = f"""
<html>
<head>
<meta charset="utf-8">
    <style>
        body {{ font-family: Arial; padding: 20px; }}
        table {{ margin-bottom: 40px; }}
      table {{
            border-collapse: collapse;
            width: 80%;
            margin: 20px auto;
            font-family: Arial, sans-serif;
            text-align: center;
        }}
        th, td {{
            border: 1px solid #dddddd;
            padding: 8px 12px;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .gpu-header {{
            font-weight: bold;
            background-color: #e9ecef;
        }}
    </style>
</head>
<body>
    {styler.to_html()}
    <br><br>
    {styler2.to_html()}
"""
#bp()
#保存其他配置参数
html += f"<li>Config.seq_input : {config.seq_len:.0f} </li>\n"
html += f"<li>Config.seq_output : {config.decode_len:.0f} </li>\n"
html += f"<li>Config.kv_cache_rate : {config.kv_cache_rate:.3f} </li>\n"
html += f"<li>Config.weights_bits : {config.weight_bits:.0f} </li>\n"
html += f"<li>Config.KVCache_bits : {config.KVCache_bits:.0f} </li>\n"
if SLOs:
      for key,val in SLOs.items():
            html += f"<li>{key} : {val:.0f} </li>\n"
#保存对比表格
html += sb.generate_gpu_comparison_html(dfs, ep_list=config.eplist,gpu_map = gpu_map,output_file="gpu_comparison.html")
html += f"""
</body>
</html>
"""

#bp()
#----输出文件名
outHTMLName = "output_report"
if User2p7T:
      outHTMLName += "_2.7T_DeepSeek"
else:
      outHTMLName += "_671B_DeepSeek"

for key in gpu_B200_decode.keys():
     outHTMLName += "_"+key
outHTMLName += "_inSeq_" + str(config.seq_len)
outHTMLName += "_outSeq_" + str(config.decode_len)
if SLOs:
      outHTMLName += "_SLO"
      for key,val in SLOs.items():
            outHTMLName += "_"+key+"_"+str(val)
outHTMLName += ".html"

with open(outHTMLName, "w") as f:
    f.write(html)
#性能对比表格
#sb.generate_gpu_comparison_html(dfs, output_file="gpu_comparison.html")

# # 查看单个GPU的性能

# In[15]:


sb.df_filter(dfs,'Rubin-NVL144',0).style\
      .bar(subset=['TPS','Total'],color='#6495ED')\
      .applymap(sb.color_positive_red, subset=['Delta'])\
      .background_gradient(subset=['Comm_Impact'],cmap=sb.cm)\
      .format(precision=3) 


# In[16]:


gpu = 'H20'
tps_limit = 1

tdf = sb.df_filter(dfs,gpu,tps_limit=tps_limit)
df_o = sb.df_sort(tdf,value='Total',ascending=False).style\
     .bar(subset=['TPS','Total'],color='#6495ED')\
     .applymap(sb.color_positive_red, subset=['Delta'])\
     .background_gradient(subset=['Comm_Impact'],cmap=sb.cm)\
     .format(precision=3) 
#bp()
"""
<class 'pandas.io.formats.style.Styler'>无法直接打印，需要添加.data
"""
print(df_o.data)
# In[ ]:

"""
=== 添加HBM read latency 数据之前
(base) bogon:~/projects/shallowsim$ python decode.py
        GPU  TP  EP  BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS      TPS_O        Total      Total_O  Comm_Impact
0  DGX-B200   4  16        128  0.186954  0.068852   0.147220  0.092353      0.013545      0.157774  0.092353  0.318539  0.184706  0.023941  20.631254  19.242685  48.470151  51.967799  6204.179351  6651.878335     0.067304
1  DGX-B200   8  16         64  0.093477  0.062220   0.097191  0.051176      0.013545      0.157774  0.051176  0.268511  0.102353 -0.008384  16.040715  16.040715  62.341359  62.341359  3989.846968  3989.846968     0.000000
2  DGX-B200   8  16         32  0.046738  0.058904   0.081096  0.030588      0.010782      0.135668  0.030588  0.227545  0.061176 -0.030701  13.514555  13.514555  73.994295  73.994295  2367.817449  2367.817449     0.000000
3  DGX-B200   8  16         16  0.023369  0.057246   0.073048  0.020294      0.009861      0.128299  0.020294  0.211208  0.040588 -0.042320  12.491883  12.491883  80.051983  80.051983  1280.831727  1280.831727     0.000000

=== 添加HBM read latency 数据之后
---------修改部分
1) def decode_time_with_ep_list(mem_read_latency=False）
2) def decode_time(mem_read_latency=False)
3) def _decode_time(mem_read_latency=False)
4) def decode_mla(mem_read_latency=False)
decode_mla部分：需要修改2个部分
4.1）load KV cache : 
      if mem_read_latency:
                load_kv_time += gpu_dict[key].get_mem_bw()
4.2）load QKV 模型参数 : def mla_elapse_time(mem_read_latency=False)
          if mem_read_latency:
            load_t += gpu.get_mem_bw()
-----不考虑读mem的延时：mem_read_latency=False: 
batch=128, (设置：当前文件：c.bs_list=[128])
卡DGX-B200, HBM，
卡数EP=16,(设置：文件shallowsim.py:函数class Config:eplist = [16])
TP=1,4,8
--不考虑 读latency，：当前文件设置：mem_read_latency=False
(Pdb) load_kv_time
0.04425834206973805
(Pdb) dense_mla, sparse_mla
(0.16893612202883287, {1: 0.2189361220288329, 4: 0.10952161220655462, 8: 0.08840459695295051})
--考虑 读latency，mem_read_latency=True
(Pdb) load_kv_time
0.04435834206973805 #验证：0.04435=0.04425+0.0001ms延时-----OK
(Pdb) dense_mla, sparse_mla
(0.16903612202883286, {1: 0.21903612202883288, 4: 0.10954661220655462, 8: 0.08841709695295051})# 增加了0.0001ms延时---验证OK

5) def decode_dense_mlp(mem_read_latency=False）
decode_dense_mlp部分
调用了t = _prefill_dense_mlp(args, gpu_dict[key], bs,mem_read_latency=mem_read_latency)
--不考虑 读latency 
decode_dense_mlp里面
(Pdb) time
0.06885210488470589
--考虑 读latency
(Pdb) time
0.06895210488470588# 总时间增加了HBM read latency 0.0001ms----OK

6）def _decode_moe_expert（）
decode MOE 部分
在 里面
--不考虑 读latency
(Pdb) load_time
0.003088235294117647
--考虑 读latency
(Pdb) load_time
0.0031882352941176467#验证，load时间增加了 0.0001ms，---OK

对比最终throughput
--不考虑 读latency
        GPU  TP  EP  BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS      TPS_O        Total      Total_O  Comm_Impact
0  DGX-B200   4  16        128  0.213194  0.068852    0.15378  0.092353      0.013545      0.157774  0.092353  0.325099  0.184706  0.017381  20.709976  19.701899  48.285907  50.756528  6180.596125  6496.835582     0.048676

--考虑 读latency
        GPU  TP  EP  BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS      TPS_O        Total      Total_O  Comm_Impact
0  DGX-B200   4  16        128  0.213394  0.068952   0.153905  0.092353      0.013745      0.160974  0.092353  0.328624  0.184706  0.017056  20.896476  19.907249  47.854958  50.232957  6125.434616  6429.818529     0.047339
TPS下降了
"""



