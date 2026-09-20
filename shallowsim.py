import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import copy as copy
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Literal
from pdb import set_trace as bp

cm = sns.light_palette("red", as_cmap=True)
NVL_GPU_LIST = [72, 144, 576]

class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class Config:
    #seq_len ,decode_len= 1660,373
    seq_len ,decode_len= 5900,499
    #seq_len ,decode_len= 103500,1100
    #seq_len ,decode_len= 4383,1210
    #seq_len ,decode_len= 8000,1000
    #decode_len = 1210
    kv_cache_rate = 0.563
    #decode_len = 1210
    bs_list = [16, 32, 64, 128, 256, 512,1024]#batch size 的可能数值
    #bs_list =[16, 32, 64, ]
    #eplist = [8, 16, 36, 72, 144, 320]#卡数量的可能数目
    eplist = [1,2,4,8,16]
    #eplist = [8,16]
    weight_bits = 8 # weight bits数
    KVCache_bits = 8
    # only consider TP=1
    #TP1 = True

# margin: profit margin; power cost: 0.08$/kWh, operation cost:0.3 of GPU card, duruance: 3 years, utility:0.4
class GPU_perf:
    def __init__(self, gpu_type, sm, comm_sm, gpu_per_node,
                 fp16_flops, fp8_flops, fp4_flops,
                 mem, mem_bw, nvlink_bw, pcie_bw, discount_rate, mem_read_latency,cost,power,margin=0.9,power_cost=0.08, operation_cost_r=0.2, durance=4,utility=0.5):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate
        self.mem_read_latency = mem_read_latency
        #硬件成本，不是采购价格
        self.cost = cost
        #硬件价格，假设90%利润率
        #bp()
        self.price = self.cost/(1-margin)
        #计算每小时成本：life durance 3 years，40% utilizaiton rate， 电费 0.08$/kWh, operation cost = 30% of hardware
        hourly_cost = self.price / (durance*365*24*utility)
        hourly_cost += operation_cost_r*hourly_cost
        hourly_cost += power_cost * power/1000
        self.hourly_cost = hourly_cost



    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate
    
    def get_mem_read_latency(self):
        return self.mem_read_latency

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate
    
    def get_cost(self):
        return self.cost

    def get_power(self):
        return self.power

def get_gpu_info(filename='./device/gpuinfo.csv',
                 discount_rate=0.85,
                 device_list=[],
                 decoding_mode=False, print_console=False):
    """Get gpu info from csv file.

    Args:
        filename (str, optional): gpu performance datasheet filepath. Defaults to './device/gpuinfo.csv'.
        discount_rate (float, optional): Estimate performance discount from Peak FLOPS and peak BW. Defaults to 0.85.
        device_list (list, optional): select dedicated gpu. Defaults to [].
        decoding_mode (bool, optional): Enable decoding mode to set comm_sm=0. Defaults to False.
        print_console (bool, optional): print result. Defaults to False.

    Returns:
        dict{GPU_perf}: gpu performance dict.
    """
    gpu_dict = {}
    df = pd.read_csv(filename)
    if print_console:
        print(df.set_index('gpu_type').to_markdown())
    if decoding_mode:
        df['comm_sm'] = 0
    #bp()
    for _, c in df.iterrows():
        key = c['gpu_type']
        gpu = GPU_perf(
            gpu_type=c['gpu_type'],
            sm=c['sm'], comm_sm=c['comm_sm'],
            fp16_flops=c['fp16'],
            fp8_flops=c['fp8'],
            fp4_flops=c['fp4'],
            mem=c['mem'],
            mem_bw=c['mem_bw'],
            nvlink_bw=c['nvlink_bw'],
            pcie_bw=c['pcie_bw'],
            gpu_per_node=c['gpu_per_node'],
            discount_rate=discount_rate,
            mem_read_latency=c['mem_read_latency'],
            cost = c['cost'],
            power = c['power'])
        if (len(device_list) == 0) | (key in device_list):
            gpu_dict[key] = gpu
    return gpu_dict,df

def gpu_category_idx(gpu_dict):
    gpu_category={}
    i = 0
    for key in gpu_dict.keys():
        gpu_category[key]=i
        i +=1
    return gpu_category

# 非吸收的版本,MLA 部分GEMM FP8 FLOP和Attention部分 FP16需要的FLOPs


def mla_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    #bp()
    """
    q_len: prefill阶段 q_len = seq_len = 4383
    q_down_proj 
    = q_len * args.dim * args.q_lora_rank  
    = 4383*7168*1536
    = 48257040384
    q_up_proj 
    = q_len * args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim) 
    = 4383*1536*128*(128+64)
    =165452709888
    kv_down_proj 
    = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim) 
    =4383*7168*(512+64)
    =18096390144
    k_up_proj 
    = kv_len * args.kv_lora_rank * \
        args.n_heads * args.qk_nope_head_dim 
    =4383*512*128*128
    =36767268864
    v_up_proj 
    = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim
    =4383*512*128*128
    =36767268864
    wo 
    = q_len * args.n_heads * args.v_head_dim * args.dim 
    =4383*128*128*7168
    =514741764096
    mha 
    = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len  # QK_score_nope
                          + q_len * kv_len * args.v_head_dim)  # ScoreV
    =128*(4383*64*4383 + 4383*128*4383 + 4383*4383*128)
    =786869821440
    (Pdb) ATTN_FP16_FLOPS (mha +wo 部分, 用FP16) 单位是 GFLOPs
    2603.223171072
    (Pdb) GEMM_FP8_FLOPS (其他纯gemm,用FP8) 单位是 GFLOPs
    590.3048209858559
    """
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_up_proj = q_len * args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = kv_len * args.kv_lora_rank * \
        args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv

    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)
    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

    # 把它看成一个标准的args.n_heads的MHA
    mha = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len  # QK_score_nope
                          + q_len * kv_len * args.v_head_dim)  # ScoreV
    wo = q_len * args.n_heads * args.v_head_dim * args.dim  # wo
    attn_sum = mha + wo
    # return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS = attn_sum * 2/1e9

    return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS

# 矩阵吸收的版本


def mla_matabsob_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate=0):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * \
        args.n_heads * args.qk_rope_head_dim  # wq_b_rope
    q_absorb = q_len * args.n_heads * (args.q_lora_rank * args.qk_nope_head_dim  # wq_b
                                       + args.qk_nope_head_dim * args.kv_lora_rank)  # w_uk

    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)  # KV-Cache命中率修正
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

    # 把它看成一个标准的args.n_heads的MQA
    mqa = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # Score_rope
                          + q_len * args.kv_lora_rank * kv_len  # Score_nope
                          + q_len * kv_len * args.kv_lora_rank)  # Score V

    attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim
    attn_sum = mqa + attn_up_proj + o_proj

    # return flops by 2* Sum(MACs)
    gemm_sum = gemm_sum * 2/1e9
    attn_sum = attn_sum * 2/1e9

    return gemm_sum + attn_sum, gemm_sum, attn_sum


def mla_mem(args: ModelArgs):
    q_down_proj = args.dim * args.q_lora_rank  # wq_a
    q_up_proj = args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv
    wo = args.n_heads * args.v_head_dim * args.dim  # wo
    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo)/1024/1024

"""
MLA部分做infer需要的时间
"""
def mla_elapse_time(args: ModelArgs,
                    gpu: GPU_perf,
                    seq_len,
                    kv_cache_rate,
                    tp=[2, 4, 8, 16, 32],
                    decoding_mode=True,
                    batchsize=1,
                    enable_gemm_fp4=True,
                    min_ar_time=0.015,  # Allreduce的静态延迟
                    mla_discount=0.7,  # based on FlashMLA result on H800
                    mla_kernel_static_time=0.05,# lauch kernel的固定时间开销
                    print_console=False,
                    mem_read_latency=False
                    ):
    #bp()
    # MLA部分的gemm和attenion部分FLOPs，区分prefill（非吸收）和decode（吸收）阶段
    if decoding_mode:
        # Decoding时计算为qlen=1, kv_cache_rate = 1
        _, gemm_flops, attn_fp16_flops = mla_matabsob_flops(
            1, seq_len, args, 1)
        gemm_flops *= batchsize#前面计算的是batchsize=1的计算量
        attn_fp16_flops *= batchsize
    else:
        # prefill阶段使用非吸收的版本，为什么此时没有考虑batchsize？？？
        _, gemm_flops, attn_fp16_flops = mla_flops(
            seq_len, seq_len, args, kv_cache_rate)
    #根据gemm和attention的FLOPs，得到计算需要时间
    """
    ====prefill
    ----计算需要时间
    此时q_len=4333=kv_len
    (Pdb) gemm_flops, attn_fp16_flops
    (590.3048209858559, 2603.223171072) 单位GFLOPs

    ---- GEMM_FP8需要时间
    MLA中GEMM部分用FP8: 所需时间 : 单位ms(因为计算量单位GFLOPS,算力单位TFLOPS)
    gemm_fp8_t = 590GFLOPs / 1427TFLOPS / 0.7
    =0.59ms

    --- ATTN_FP16需要时间
    MLA中attention部分用FP16计算: 所需时间
    attn_fp16_t = 2603FLOPs / 713TFLOPS / 0.7
    = 5.2ms

    ---内存需要时间
    访存量 mla_mem(args) = 178 MB # 单位是MBbyte
    H800的DRAM带宽BW gpu.get_mem_bw()=3400 GB/s #单位是GBytes/s
    所以结果的单位是ms
    load_t = mla_mem(args) / gpu.get_mem_bw()，
    = 178 MBytes / 3400 GBytes/s
    =0.05248 ms

    ---所以prefill阶段, 因为seq大, 所以是计算密集的 : 计算时间比访存长很多

    ---allReduce通信时间
    TP引起的allReduce长度（token 长度）在prefill阶段=seq_len=4383
    每个token对应allReduce的信息量就是dim （就是激活），假设FP16，每个信息2bytes
    allReduce的通信量
    =all_reduce_comm_size 
    = ar_len * args.dim * 2 / 1024/1024
    = 4383*7168*2/1024/1024 单位 MBbytes
    = 59.9MB

    allReduce需要时间
    =all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time
    = 59.9M bytes / 170GB/s + 0.015ms
    = 0.35 + 0.015 ms
    = 0.367ms

    """
    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops() / mla_discount
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops() / mla_discount

    # load weight，根据MLA部分内存读写量，计算内存加载时间===？？？没有考虑HBM latency，仅仅考虑了BW
    # 这部分prefill阶段，需要加载所有参数，但是decode阶段，用KV cache了，还需要加载和QKV所有模型参数吗？
    load_t = mla_mem(args) / gpu.get_mem_bw() 
    #bp()
    if mem_read_latency:
        load_t += gpu.get_mem_read_latency()
    #总时间 = gemm时间+attention时间 +内存加载时间
    total = gemm_fp8_t + attn_fp16_t + load_t

    if enable_gemm_fp4:
        if gpu.get_fp4_flops() == 0:
            if print_console:
                print('[%8s]This GPU does not support FP4' % gpu.gpu_type)
        else:
            gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
            #====??? 为什么没有考虑 load_t 
            #total = gemm_fp4_t + attn_fp16_t
            total = gemm_fp4_t + attn_fp16_t + load_t

    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024/1024  # fp16 take 2Bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    """
    ---不考虑TP (TP=1) 的总时间：
    total = gemm时间 + attn时间+ 内存加载时间
    = 0.5908394411589774+ 5.213796214868718+0.052481617647058824
    =5.857 ms

    ---考虑TP时间：每个卡时间降低，同时增加allReduce时间
    v=tp数：
    tp情形时间 ，如v=4: TP4
    tp_time[v]
    = 没有TP情形计算和内存需要时间 / TP数 + allReduce时间 + mla固定时间
    = 5.875/4 ms + 0.367ms + 0.05ms
    = 1.88ms
    """
    tp_time = {}
    for v in tp:
        if v == 1:
            tp_time[v] = total + mla_kernel_static_time
        else:
            tp_time[v] = total / v + all_reduce_t + mla_kernel_static_time

    if print_console:
        if enable_gemm_fp4 & (gpu.get_fp4_flops() != 0):
            print("[%8s]GEMM_FP4 Elapsed time(ms): %.3f" %
                  (gpu.gpu_type, gemm_fp4_t))
        print("[%8s]GEMM_FP8 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, gemm_fp8_t))
        print("[%8s]ATTN_FP16 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, attn_fp16_t))
        print("[%8s]Total Elapsed time(ms):%.3f" % (gpu.gpu_type, total))
        print("[%8s]AR Elapsed time(ms):%.3f" % (gpu.gpu_type, all_reduce_t))
        for v in tp:
            print("[%8s]TP[%2d] Elapsed time(ms):%.3f" %
                  (gpu.gpu_type, v, tp_time[v]))

    return total, tp_time


def prefill_mla(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP1', 'TP4', 'TP8'])
    for key in gpu_dict.keys():
        tp1, tp_list = mla_elapse_time(args, gpu_dict[key],
                                       seq_len, kv_cache_rate,
                                       tp=[4, 8],
                                       decoding_mode=False,
                                       enable_gemm_fp4=True,
                                       print_console=print_console)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, tp1] + \
            list(tp_list.values())
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

"""
denseMLP 的FLOPs
这部分的参数量是
3  * args.dim * args.inter_dim
= 3*7168*18432
= 396361728 
= 396.36 M
1个 token 计算量
= 1 * 2* 396.36 / 1000
= 0.79272 GFLOPs
长度为 4383的 seq需要的计算量
4383*0.79272 GFLOPs
= 3474.5 GFLOPs

这里应该 / 1024**3，不应该 1e9？？？
"""
def densmlp_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.inter_dim * 2/1e9

"""
dense MLP 部分的参数容量
=3*7168*18432/1024/1024
=378 MBbyte
"""
def densmlp_mem(args: ModelArgs):
    return 3 * args.dim * args.inter_dim / 1024/1024

"""
dense MLP 部分prefill 需要的时间
这部分为何没有考虑TP====？？？
"""
def _prefill_dense_mlp(args: ModelArgs, gpu: GPU_perf, seq_len, print_console=False,mem_read_latency=False):
    """
    ------计算需要时间
    gemm_flops = 3474.5GFLOPs
    gpu.get_fp8_flops()=1427 TFLOPS
    gemm_time = 3474.5GFLOPs / 1427TFLOPS
    = 2.434ms
    ------访存需要时间
    densmlp_mem(args) = 378M Bytes
    gpu.get_mem_bw() = 3400 Gbyte/s
    load_time = 378/3400 = 0.11 ms

    总时间
    gemm_time + load_time
    = 2.54ms
    """
    #bp()
    gemm_flops = densmlp_flops(args, seq_len)
    if gpu.get_fp4_flops() != 0:
        gemm_time = gemm_flops / gpu.get_fp4_flops()
    else:
        gemm_time = gemm_flops / gpu.get_fp8_flops()

    load_time = densmlp_mem(args) / gpu.get_mem_bw()
    if mem_read_latency:
        load_time += gpu.get_mem_read_latency()

    gemm_time = gemm_time + load_time
    if print_console:
        print("[%8s]Elapsed time(ms): %.3f" % (gpu.gpu_type, gemm_time))
    return gemm_time


def prefill_dense_mlp(args: ModelArgs, gpu_dict, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'DenseMLP'])
    for key in gpu_dict.keys():
        t = _prefill_dense_mlp(args, gpu_dict[key], seq_len, print_console=print_console)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, t]
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def moe_expert_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.moe_inter_dim * 2/1e9


def moe_expert_mem(args: ModelArgs):
    return 3 * args.dim * args.moe_inter_dim / 1024 / 1024

"""
MOE部分, prefill阶段需要的时间
"""
def _prefill_moe(args: ModelArgs, gpu: GPU_perf, seq_len, tp, dp):
    #bp()
    """
    计算的都是分配到1个GPU卡上的 tokens,或者需要读入的参数容量

    -----加载参数需要时间
    1个moe 加载内存量 moe_expert_mem(args) =44MB : 访问内存数据, 单位 MB
    H800 DRAM带宽 gpu.get_mem_bw()=3400GB/s (考虑0.85利用率), 带宽 单位 GB/s
    加载参数所需时间:load_time = 44/3400 = 0.01235 ms 单位是ms : 因为上下相差3个数量级

    
    1个卡的算力:
    gemm_flops=1979 * 0.85*(132-20)/132 = 1427 TFLOPS//132个SM,20个做通信,剩余112做计算
    num_device = 4*8=32 : 32个卡做expert计算
    num_shared_token = dp * seq_len / num_device
    = 8 *4383/32 = 1095 
    # 共享的token长度 为什么这么计算===??? : 其实这里TP按照行切分,所以输入token
    长度需要 4383/4=1095 tokens
    shared_flops = 3*seq_len * args.dim * args.moe_inter_dim * 2/1e9
    = 2*3*
    = 96 GFLOPs : 计算量 单位 GFLOPs
    H800 FP8 算力是 1427 TFLOPS : 卡算力单位是 TFLOPS
    这部分计算需要的时间：
    96GFLOPs / 1427 TFLOPS = 0.067 ms 单位是ms: 因为上下相差3个数量级
    #是1个共享专家的计算量
    
    -----共享部分计算需要时间
    =shared_time
    = 共享FLOPs/计算能力 + 读一个共享专家参数需要时间 * 专家个数1
    = 0.067ms + 0.01235ms = 0.0799ms

    args.n_activated_experts=8
    num_routed_token = seq_len * dp * args.n_activated_experts / num_device
    1张卡上需要处理的路由tokens数量
    = 输入seq长度/tp行切分 * 每个token需要激活的专家数目
    = 4383/4*8=8766

    路由tokens需要的计算量
    =单个专家计算量(路由tokens数量)
    = 2*3*8766*7168*2048 / 1e9 GFLOPs
    = 772 GFLOPs

    每1个GPU卡上, 需要load多少个专家的参数 
    expert_num
    =总专家数 / 总卡数
    =256/(32) = 8 ===??? 为什么不需要加载所有专家参数===???

    ------1个GPU卡上,处理路由tokens需要时间
    =routed_time
    = 1个卡上处理分配给本卡的tokens计算时间 + 1个卡上加载1个专家参数时间*本卡需要加载专家个数
    = flops / 1卡计算能力 + load_time * expert_num
    = 772GFLOPs / 1427TFLOPS + 0.01235*8
    = 0.54 + 0.098
    =0.6397 ms
    """
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    gemm_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    num_device = tp * dp
    num_shared_token = dp * seq_len / num_device
    shared_flops = moe_expert_flops(args, num_shared_token)
    shared_time = shared_flops / gemm_flops + load_time

    num_routed_token = seq_len * dp * args.n_activated_experts / num_device
    routed_flops = moe_expert_flops(args, num_routed_token)
    #此处存疑 ????
    expert_num = math.ceil(args.n_routed_experts) / dp / tp
    routed_time = routed_flops / gemm_flops + load_time * expert_num
    
    return shared_time, routed_time

# 1个卡上, prefill 阶段，MOE部分需要的时间
def prefill_moe(args: ModelArgs, gpu_dict, seq_len,
                tp_list=[4, 8],
                dp_list=[4, 8, 9],
                print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'DP',
                      'Shared Expert', 'Routed Expert'])
    for key in gpu_dict.keys():
        for tp in tp_list:
            for dp in dp_list:
                s, r = _prefill_moe(args, gpu_dict[key], seq_len, tp, dp)
                df.loc[len(df)] = [gpu_dict[key].gpu_type, tp, dp, s, r]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        df['DP'] = df['DP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

# prefill 阶段all2all通信所需时间
def _prefill_alltoall(args: ModelArgs, gpu, seq_len, tp, static_latency=0.05):
    """
    每个node有8卡，如果TP=4，那么DP=2
    MoE gating之后一个token最多被分发到4个卡上
    分发数据量 
    ===??? 感觉这里不应该 / gpu_per_node ----？？？
    dispatch_size 
    = （seq长度 * dp个数）个token数 * 每个token有dim个数据 * 每个token会被activated个专家
    处理 * 每个专家处理完token会发到其他4-1个卡上从而引起卡间all2all通信
    = seq_len * dp * (dispatch_node -1) * n_activated_experts * dim / gpu_per_node
    = 179 MByte
    如果支持FP4: 分发数据量 /2 = 89MB
    聚合带宽
    comm_bw 
    = gpu.get_pcie_bw() * gpu.gpu_per_node
    = 42.5GB/s * 8
    = 340GB/s

    -----分发需要时间
    =dispatch_size / comm_bw
    =89MB / 340GB/s
    = 0.26 ms

    -----combine聚合需要时间
    聚合的数据量
    = 从分发的FP8变成FP16，所以*2
    = 359 M bytes

    聚合通信需要时间
    = combine_size/comm_bw
    = 359M Bytes / 340GB/s
    = 1.1 ms
    """
    #bp()
    if gpu.gpu_per_node == 8:
        dp = gpu.gpu_per_node/tp
        dispatch_node = 4
        dispatch_size = (dispatch_node - 1) * dp * seq_len * \
            args.n_activated_experts / gpu.gpu_per_node * args.dim / 1024/1024
        comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
    else:
        # NVL72
        expert_num = math.ceil(args.n_routed_experts / gpu.gpu_per_node)
        dispatch_prob = (args.n_routed_experts - expert_num) / \
            args.n_routed_experts
        dispatch_size = dispatch_prob * args.n_activated_experts * \
            seq_len/tp * args.dim / 1024/1024
        comm_bw = gpu.get_nvlink_bw()

    combine_size = 2 * dispatch_size  # fp16
    if gpu.get_fp4_flops != 0:
        #如果支持FP4，dispatch阶段传输 4bits 信息，不是1 byte
        dispatch_size = dispatch_size / 2
    dispatch_time = dispatch_size / comm_bw + static_latency
    combine_time = combine_size / comm_bw + static_latency
    return dispatch_time, combine_time


def prefill_alltoall(args: ModelArgs, gpu_dict, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'Dispatch', 'Combine'])
    for tp in [4, 8]:
        for key in gpu_dict.keys():
            dispatch_time, combine_time = _prefill_alltoall(
                args, gpu_dict[key], seq_len, tp)
            df.loc[len(df)] = [key, tp, dispatch_time, combine_time]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _prefill_time(args: ModelArgs, gpu, seq_len, kv_cache_rate, tp, dp):
    #bp()
    dense_mla, tp_mla = mla_elapse_time(args, gpu,
                                        seq_len, kv_cache_rate,
                                        tp=[tp],
                                        decoding_mode=False,
                                        enable_gemm_fp4=True)
    dense_mlp = _prefill_dense_mlp(args, gpu, seq_len)
    shared, routed = _prefill_moe(args, gpu, seq_len, tp, dp)
    dispatch, combine = _prefill_alltoall(args, gpu, seq_len, tp)
    """
    prefill阶段，给定seq长度，推理模型一层的各个模块需要时间
    (Pdb) dense_mla, dense_mlp, tp_mla[tp], 
    shared, combine, routed, dispatch
(5.8571172736747545, 2.5455340306916745, 1.8817724250363357, 
0.07997398451267723, 1.1074793198529411, 0.6397918761014179, 0.31436982996323526)
    mla时间 5 ms
    mlp时间 2.5 ms
    shared专家时间 0.08ms
    路由专家时间 0.64ms
    all2all通信时间：分发时间FP8/FP4： 0.52ms / 0.26ms
    all2all通信时间：聚合时间FP16：1.1ms 
    """
    return dense_mla, dense_mlp, tp_mla[tp], shared, combine, routed, dispatch

# 计算prefill 阶段，整个model的推理时间
def prefill_time(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate, tp, dp, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'MLA', 'DenseMLP', 'TP_MLA', 'Shared Expert',
                      'Combine', 'Overlap1', 'Routed Expert', 'Dispatch', 'Overlap2'])
    df2 = pd.DataFrame(columns=['GPU', 'Compute', 'Comm', 'Sum'])
    n_sparse_layers = args.n_layers - args.n_dense_layers
    df.loc[len(df)] = ['Layers', args.n_dense_layers, args.n_dense_layers,  # MLA+ DenseMLP
                       n_sparse_layers, n_sparse_layers, n_sparse_layers, n_sparse_layers,
                       n_sparse_layers, n_sparse_layers, n_sparse_layers]
    bp()
    """
    (Pdb) dense_mla, dense_mlp, tp_mla, 
    shared, combine, routed, dispatch
    (5.8571172736747545, 2.5455340306916745, 1.8817724250363357, 
    0.07997398451267723, 1.1074793198529411, 0.6397918761014179, 0.31436982996323526)

    ---1个DP完成4383长度prefill任务需要的时间：
    不考虑overlap 
    为何3个dense层 MLA不考虑TP？ 58个MoE层MLA考虑TP===???
    3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert +Dispatch + Combine)
    =3*(2.5 + 5.86) + 58*(1.88+ 0.08 + 0.64 + 0.3 + 1.1)
    = 257 ms

    考虑overlap
    3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert)
    = 3*( 2.5 + 5.85) + 58*(1.88 + 0.08 + 0.64)
    = 175 ms

    DP=8，TP=4，seq_len=4383, 节点数=4，那么
    ----单节点的吞吐
    = 所有节点DP总吞吐 / 节点数
    = DP数*1个DP 1s完成的tokens数 / 节点数
    = （DP * 4383tokens每个DP * 1000ms / 1个DP完成4383tokens推理时间） / 节点数
    = 8* 4383*1000/175/4
    =50009 tokens/s/节点

    官方数据 73000tokens/s/节点 （含有缓存命中），如果不命中大约 5万
    """
    for key in gpu_dict.keys():
        dense_mla, dense_mlp, tp_mla, shared, combine, routed, dispatch = _prefill_time(
            args, gpu_dict[key], seq_len, kv_cache_rate, tp, dp)
        overlap1 = combine - (tp_mla + shared)
        overlap2 = dispatch - routed
        df.loc[len(df)] = [key, dense_mla, dense_mlp, tp_mla, shared,
                           combine, overlap1, routed, dispatch, overlap2]
        comp_time = args.n_dense_layers * \
            (dense_mla + dense_mlp) + n_sparse_layers * (tp_mla + shared + routed)
        comm_time = n_sparse_layers * (combine + dispatch)
        sum_time = comp_time
        if overlap1 > 0:
            sum_time += overlap1 * n_sparse_layers
        if overlap2 > 0:
            sum_time += overlap2 * n_sparse_layers
        df2.loc[len(df2)] = [key, comp_time, comm_time, sum_time]
    df = df.set_index('GPU').T
    df2 = df2.set_index('GPU').T
    if print_console:
        df['Layers'] = df['Layers'].astype(int).astype(str)
        print(df.to_markdown(floatfmt=".3f"))
        print('-----------SUM-------------')
        print(df2.to_markdown(floatfmt=".3f"))
    return df, df2

# Decoding
"""
decode阶段能处理的最大batch：
先计算：decode阶段，每个batch=1，每个GPU卡，需要存储的kv cache的容量是多大
再计算：根据1个GPU卡容量,  去掉其他部分还剩余的容量，除以这个size，得到最大batch size

---每1个batch需要的 （为什么需要*TP？？？）
kv cache
=(输入seq长度+输出seq长度)*(kv_lora_rank+qk_rope_head_dim)*网络layer数*TP数
=(4383+1210)*(512+64)*61*1
=196515648 个数值

---1个GPU的可以分配给KV cache的容量
= GPU容量*利用率 - 模型参数外需要的容量 - 一层mla部分参数*层数/tp
= 80GB*0.9 - 2.91GB -  187.17MB*61/1/1024 GB
= 57.94GB

再去掉 分配到1个卡上专家参数：假设每层MoE分配33个专家给一个卡
57.94GB -  44.05MB*(61-3)*33/1024
= -24.4GB

出现负数怎么处理？？？

mem * 1024 * 1024 * 1024 / kv_cache
=-133 batch

如果分配到1个卡上专家个数expert_num很小：比如
>>> 57.94 -  44.05*(61-3)*23/1024
0.5545507812500077
:当一个卡上专家个数=23而不是33时：这个卡上就有剩余空间放kv cache，进而可以计算batch大小
"""
def _decoding_batchsize(args: ModelArgs, gpu: GPU_perf, seq_len, decode_len, tp, expert_num):
    mem_util_rate = 0.9  # torch/activation等其它开销的折扣
    # 假设weight和KV都是 8 bits
    mla = 187.17  # MLA的参数(单位M)
    expert_mem = 44.05  # expert的参数(单位M)
    others_parameter = 2.91  # 其它参数2.91GB
    kv_cache = (seq_len+decode_len) * (args.kv_lora_rank +
                                       args.qk_rope_head_dim) * args.n_layers * tp
    #bp()
    #---如果weight，KV 可以 4 bits
    if Config.weight_bits == 4:
        mla /= 2
        expert_mem /= 2
        others_parameter /=2 
    if Config.KVCache_bits == 4:
        kv_cache /= 2

    mem = gpu.mem * mem_util_rate - others_parameter - mla * args.n_layers/tp/1024
    mem -= expert_mem * \
        (args.n_layers - args.n_dense_layers) * expert_num / 1024
    return mem * 1024 * 1024 * 1024 / kv_cache


def decode_batchsize(args: ModelArgs, gpu_dict, seq_len, decode_len, tp):
    df = pd.DataFrame(columns=['GPU', 'EP320', 'EP144', 'EP72', 'EP34'])
    for key in gpu_dict.keys():
        item = key
        value = [item]
        for exp_num in [2, 3, 5, 9]:
            bs = _decoding_batchsize(
                args, gpu_dict[key], seq_len, decode_len, tp, exp_num)
            #bp()
            value.append(bs)
        df.loc[len(df)] = value
    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))
    return df

"""
decode阶段，MLA部分需要时间
dense_mla, sparse_mla 分别是MLA部分：TP=1时MLA时间， TP大于1时MLA需要的时间
    GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA
0  H800        16   1  0.011065  0.094070   0.144070 ---此时应该dense=sparse？？？因为TP本来就是1？
1  H800        16   4  0.011065  0.094070   0.089804
2  H800        16   8  0.011065  0.094070   0.078046
3  H800        32   1  0.022129  0.135659   0.185659
4  H800        32   4  0.022129  0.135659   0.101488
5  H800        64   1  0.044258  0.218836   0.268836
"""
def decode_mla(args: ModelArgs, gpu_dict, bs_list, seq_len, decode_len, expert_num=2, print_console=False,mem_read_latency=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'LoadKV', 'DenseMLA', 'SparseMLA'])
    tp_list = [1, 4, 8]
    for key in gpu_dict.keys():
        for bs in bs_list:
            #bp()
            kv_cache = seq_len * (args.kv_lora_rank +
                                  args.qk_rope_head_dim) * bs
            # 没有考虑latency？？？,单位是ms
            load_kv_time = kv_cache / 1024/1024 / 1024 / gpu_dict[key].get_mem_bw() * 1000
            if mem_read_latency:
                load_kv_time += gpu_dict[key].get_mem_read_latency()
            dense_mla, sparse_mla = mla_elapse_time(args, gpu_dict[key],
                                                    seq_len, kv_cache_rate=1,
                                                    tp=tp_list,
                                                    batchsize=bs,
                                                    decoding_mode=True,
                                                    enable_gemm_fp4=True,
                                                    mem_read_latency=mem_read_latency)
            #bp()
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num,
                                       load_kv_time, dense_mla, sparse_mla[tp_num]]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

"""
decode阶段
计算dense MLP 的时间
(Pdb) dense_mlp

"""
def decode_dense_mlp(args: ModelArgs, gpu_dict, bs_list, seq_len, decode_len, expert_num=2, print_console=False,mem_read_latency=False):
    tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize', 'TP', 'DenseMLP','HourlyCost'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            # MLP 部分，prefill 和decode 阶段没有区别
            t = _prefill_dense_mlp(args, gpu_dict[key], bs,mem_read_latency=mem_read_latency)
            #bp()
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:#如果 最大能支持的bs 《 这个batch size ，就忽略
                    continue
                else:
                    #df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num, t]
                    #添加hourly_cost
                    #bp()
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num, t,gpu_dict[key].hourly_cost]

    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'DenseMLP']
                                ].set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def n_pow2_range(n:int):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n+1
    return n
"""
decode阶段，计算 shared 和 routed 专家需要的时间

"""
def _decode_moe_expert(args: ModelArgs, gpu: GPU_perf, bs, 
                       gemm_group_per_device, device_num,mem_read_latency=False):
    #没有考虑latency？？？
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    if gpu.get_fp4_flops() != 0:
        load_time = load_time /2

    if mem_read_latency:
        load_time += gpu.get_mem_read_latency()
    #bp()
    gpu_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    
    total_expert = gemm_group_per_device * device_num
    m_per_group = bs * args.n_activated_experts * device_num / total_expert
   
    '''
    # TODO: 
    # 基于 group_gemm num 和 m_per_group 调整折扣因子
    # 可以基于Profile实测结果查表, 并将数据放在GPU_Perf结构题中
    # 此处简化以m_per_group估计如下
    '''

    #data from hs's profiling result
    flops_discounts = {
        1: 0.05,
        2: 0.05,
        4: 0.05,
        8: 0.05,
        16: 0.08,
        32: 0.1,
        64: 0.2,
        128: 0.35,
        256: 0.4,
        512: 0.6,
        1024: 0.7,
        2048: 0.7,
        4096: 0.7,
        8192: 0.7,
        16384: 0.7,
        32768: 0.7,
        65536: 0.7
    }

    # H20 exception based on hs's result
    if gpu.gpu_type.find('H20')!= -1 :
        flops_discounts = {
        1: 0.06,
        2: 0.06,
        4: 0.06,
        8: 0.12,
        16: 0.25,
        32: 0.45,
        64: 0.8,
        128: 0.9,
        256: 1.0,
        512: 1.0,
        1024: 1.0,
        2048: 1.0,
        4096: 1.0,
        8192: 1.0,
        16384: 1.0,
        32768: 1.0,
        65536: 1.0
    }

    gpu_flops = gpu_flops * flops_discounts[n_pow2_range(int(m_per_group))]
    
    shared_flops = moe_expert_flops(args, bs)
    shared_time = shared_flops / gpu_flops + load_time

    num_routed_token = bs * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu_flops + load_time * gemm_group_per_device
    return shared_time, routed_time

"""
decode阶段，moe部分时间
mbs 什么含义？？？
"""
def decode_moe_expert(args: ModelArgs, gpu_dict, 
                      bs_list, seq_len, decode_len, 
                      gemm_group_per_device,
                      device_num,
                      mbs=2, 
                      print_console=False,mem_read_latency=False):
    tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'SharedExpert', 'RoutedExpert'])
    for gpu_key in gpu_dict.keys():
        for bs in bs_list:
            s, r = _decode_moe_expert(
                args, gpu_dict[gpu_key], bs/mbs, 
                gemm_group_per_device=gemm_group_per_device, 
                device_num=device_num,mem_read_latency=mem_read_latency)
            s *= mbs
            r *= mbs
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[gpu_key], seq_len, decode_len, 
                    expert_num= gemm_group_per_device+1, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[gpu_key].gpu_type,
                                       str(bs), tp_num, s, r]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'SharedExpert',
              'RoutedExpert']].set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _moe_a2a(args: ModelArgs, gpu: GPU_perf, bs, expert_num, device_num, fp8_combine=False, static_latency=0.005, mbs=2):
    dispatch_size = bs * args.dim * args.n_activated_experts / 1024/1024
    if fp8_combine & (gpu.get_fp4_flops() != 0):  # 支持FP4GPU才能开启FP8 Combine
        combine_size = dispatch_size
    else:
        combine_size = dispatch_size * 2  # FP16
    if gpu.gpu_per_node == 8:
        comm_bw = gpu.get_pcie_bw()
        # single host deployment
        if args.n_routed_experts / (expert_num - 1) == gpu.gpu_per_node:
            comm_bw = gpu.get_nvlink_bw()
    #NVL72 /144 / 576
    elif (gpu.gpu_per_node in NVL_GPU_LIST) & (device_num >  gpu.gpu_per_node):
            comm_bw = gpu.get_pcie_bw()
    else:
        comm_bw = gpu.get_nvlink_bw()

    dispatch_t = dispatch_size / comm_bw + static_latency * mbs
    combine_t = combine_size / comm_bw + static_latency * mbs
    return dispatch_t, combine_t


def decode_a2a(args: ModelArgs, gpu_dict,
               bs_list, seq_len, decode_len,
               expert_num, device_num,
               mbs=2,
               print_console=False, fp8_combine=False):
    tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'Dispatch', 'Combine'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            dispatch_time, combine_time = _moe_a2a(
                args, gpu_dict[key], bs, 
                expert_num=expert_num, device_num=device_num, 
                mbs=mbs, fp8_combine=fp8_combine)
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], 
                    seq_len, decode_len, 
                    expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs,
                                       tp_num, dispatch_time, combine_time]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'Dispatch', 'Combine']].set_index(
            'GPU').to_markdown(floatfmt=".3f"))
    return df

"""
计算decode阶段，1层网络，需要的时间
gpu='H800'
bs_list=[16, 32, 64, 128, 256, 512, 1024, 2048]
seq_len,decode_len = 4383,1210
gemm_group_per_device=256/8卡=32 group/卡
device_num=8卡
mbs = 2??? 什么意思？？？
expert_per_device ==8+1共享专家 = 9个专家

参数
bs_list, seq_len, decode_len,gemm_group_per_device,
=
(batch选择：[16, 32, 64, 128, 256, 512, 1024, 2048], 输入：4383, 输出：1210)，16个卡

mla:
    GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA
0  H800        16   1  0.011065  0.094070   0.144070
1  H800        16   4  0.011065  0.094070   0.089804
2  H800        16   8  0.011065  0.094070   0.078046
3  H800        32   1  0.022129  0.135659   0.185659
4  H800        32   4  0.022129  0.135659   0.101488
5  H800        64   1  0.044258  0.218836   0.268836

dense_mlp:
    GPU BatchSize  TP  DenseMLP
0  H800        16   1  0.118717
1  H800        16   4  0.118717
2  H800        16   8  0.118717
3  H800        32   1  0.126257
4  H800        32   4  0.126257
5  H800        64   1  0.141337

moe:
(Pdb) moe
    GPU BatchSize  TP  SharedExpert  RoutedExpert
0  H800        16   1      0.041462      0.529340
1  H800        16   4      0.041462      0.529340
2  H800        16   8      0.041462      0.529340
3  H800        32   1      0.045651      0.562852
4  H800        32   4      0.045651      0.562852
5  H800        64   1      0.058217      0.663386

a2a
(Pdb) a2a
    GPU BatchSize  TP  Dispatch   Combine
0  H800        16   1  0.030588  0.051176
1  H800        16   4  0.030588  0.051176
2  H800        16   8  0.030588  0.051176
3  H800        32   1  0.051176  0.092353
4  H800        32   4  0.051176  0.092353
5  H800        64   1  0.092353  0.174706

最后返回合并数据
(Pdb) df
    GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA  DenseMLP  SharedExpert  RoutedExpert  Dispatch   Combine
0  H800        16   1  0.011065  0.094070   0.144070  0.118717      0.041462      0.529340  0.030588  0.051176
1  H800        16   4  0.011065  0.094070   0.089804  0.118717      0.041462      0.529340  0.030588  0.051176
2  H800        16   8  0.011065  0.094070   0.078046  0.118717      0.041462      0.529340  0.030588  0.051176
3  H800        32   1  0.022129  0.135659   0.185659  0.126257      0.045651      0.562852  0.051176  0.092353
4  H800        32   4  0.022129  0.135659   0.101488  0.126257      0.045651      0.562852  0.051176  0.092353
5  H800        64   1  0.044258  0.218836   0.268836  0.141337      0.058217      0.663386  0.092353  0.174706
"""
def _decode_time(args: ModelArgs, gpu,
                 bs_list, seq_len, decode_len,
                 gemm_group_per_device,
                 device_num,
                 mbs=2,
                 fp8_combine=False,
                 print_console=False,
                 mem_read_latency=False):
    #bp()
    expert_per_device = gemm_group_per_device + 1  # add shared expert
    mla = decode_mla(args, gpu, bs_list, seq_len,
                     decode_len, expert_num=expert_per_device,mem_read_latency=mem_read_latency)
    dense_mlp = decode_dense_mlp(
        args, gpu, bs_list, seq_len, decode_len, expert_num=expert_per_device,mem_read_latency=mem_read_latency)
    moe = decode_moe_expert(args, gpu, bs_list, seq_len,
                            decode_len, mbs=mbs,
                            gemm_group_per_device=gemm_group_per_device,
                            device_num=device_num,mem_read_latency=mem_read_latency)
    a2a = decode_a2a(args, gpu, bs_list, seq_len, decode_len,
                     expert_num=expert_per_device, device_num= device_num,
                     fp8_combine=fp8_combine, mbs=mbs)
    dfs = [mla, dense_mlp, moe, a2a]

    for decode_df in dfs:
        decode_df['BatchSize'] = decode_df['BatchSize'].astype(int).astype(str)
    df = reduce(lambda left, right: pd.merge(left, right, on=[
                'GPU', 'BatchSize', 'TP'], how='left'), dfs)
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    #bp()
    return df

"""
计算解码decode阶段时间，所有layers，不仅仅1层
"""
def decode_time(args: ModelArgs, gpu_dict,
                bs_list, seq_len, decode_len,
                gemm_group_per_device,
                device_num,
                tps_limit=0,
                fp8_combine=False,
                print_console=False,
                mem_read_latency=False):

    df = _decode_time(args, gpu_dict, bs_list, seq_len, decode_len,
                      gemm_group_per_device=gemm_group_per_device,
                      device_num=device_num,
                      fp8_combine=fp8_combine,
                      mem_read_latency=mem_read_latency)

    def overlap_adjust(r):
        if r['Delta'] > 0:
            return r['TPOT_O'] + r['Delta'] * (args.n_layers - args.n_dense_layers)
        else:
            return r['TPOT_O']

    # 修正TP执行时间, 按照加载FP8的KV计算
    # dense 层attention部分
    """
    """
    #bp()
    """
    1层时间：denseMLA 时间 = denseMLA （TP=1情况）计算时间 + 加载KV 时间； sparseMLA（TP >1）计算时间也同理
    如
    3  H800        32   1  0.022129  0.135659   0.185659  0.126257      0.045651      0.562852  0.051176  0.092353
    之后变成
    3  H800        32   1  0.022129  0.157788   0.185659  0.126257      0.045651      0.562852  0.051176  0.0
    因为：0.157788= 0.022129 + 0.135659===OK
    
    """
    df['DenseMLA'] = df['DenseMLA'] + df['LoadKV']
    df['SparseMLA'] = df['SparseMLA'] + df['LoadKV']
    # 1层计算时间 = SparseMLA时间 + 共享专家时间 + 路由专家时间 ，
    """
    之后得到新的一列COMP_SUM
        GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA  DenseMLP  SharedExpert  RoutedExpert  Dispatch   Combine  COMP_SUM
        。。。
    3  H800        32   1  0.022129  0.157788   0.207788  0.126257      0.045651      0.562852  0.051176  0.092353  0.816290
    。。。
    (Pdb) 0.207788 +   0.045651   +   0.562852 
    0.816291===OK
    """
    df['COMP_SUM'] = df['SparseMLA'] + df['SharedExpert'] + df['RoutedExpert']
    """
    之后得到
        GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA  DenseMLP  SharedExpert  RoutedExpert  Dispatch   Combine  COMP_SUM  COMM_SUM
    3  H800        32   1  0.022129  0.157788   0.207788  0.126257      0.045651      0.562852  0.051176  0.092353  0.816290  0.143529
    验算：
    (Pdb) 0.051176  +0.092353
    0.14352900000000002====OK
    """
    # 1层通信时间 = a2a分发时间 + a2a合并时间
    df['COMM_SUM'] = df['Dispatch'] + df['Combine']
    # ？？？：delta = 通信时间 -计算时间
    df['Delta'] = df['COMM_SUM'] - df['SparseMLA'] - df['SharedExpert']
    # decode阶段，token间隔时间 = dense层数 * （dense层MLA时间 + dense层MLP时间）
    """
        GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA  DenseMLP  SharedExpert  RoutedExpert  Dispatch   Combine  COMP_SUM  COMM_SUM     Delta    TPOT_O
    3  H800        32   1  0.022129  0.157788   0.207788  0.126257      0.045651      0.562852  0.051176  0.092353  0.816290  0.143529 -0.109909  0.852134
    验算：
    (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers
    =(0.157788  +  0.126257)*3
    0.852135====OK

    之后再加sparse层

        GPU BatchSize  TP    LoadKV  DenseMLA  SparseMLA  DenseMLP  SharedExpert  RoutedExpert  Dispatch   Combine  COMP_SUM  COMM_SUM     Delta    TPOT_O
    3  H800        32   1  0.022129  0.157788   0.207788  0.126257      0.045651      0.562852  0.051176  0.092353  0.816290  0.143529 -0.109909  48.196983
    验算
    dense层时间 + (sparseMLA+sharedExpert+routedExpert)*58
    =0.852135+（0.207788  +      0.045651    +  0.562852）*58
    =48.19689699999999 ===OK
    """
    df['TPOT_O'] = (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers
    # token间隔时间 还要再加上 += sparse层数 *（sparseMLA时间+共享专家时间 + 路由专家时间）
    df['TPOT_O'] += (df['SparseMLA'] + df['SharedExpert'] +
                     df['RoutedExpert']) * (args.n_layers - args.n_dense_layers)
    # 对TPOT做一个调整
    df['TPOT'] = df.apply(lambda row:  overlap_adjust(row), axis=1)
    #删除不需要的列
    df = df[['GPU', 'TP', 'BatchSize', 'DenseMLA', 'DenseMLP', 'SparseMLA', 'Combine',
             'SharedExpert', 'RoutedExpert', 'Dispatch', 'COMP_SUM', 'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O','HourlyCost']]
    # 'TPS' 这个数值是token per second，每秒token数  = 1000ms / token间隔ms数值,是一个batch的TPS
    """
    (Pdb) df
    GPU  TP BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS
0  H800   1        16  0.105135  0.118717   0.155135  0.051176      0.041462      0.529340  0.030588  0.725937  0.081765 -0.114832  42.775888  42.775888  23.377656
1  H800   4        16  0.105135  0.118717   0.100869  0.051176      0.041462      0.529340  0.030588  0.671671  0.081765 -0.060566  39.628464  39.628464  25.234387
2  H800   8        16  0.105135  0.118717   0.089110  0.051176      0.041462      0.529340  0.030588  0.659912  0.081765 -0.048807  38.946455  38.946455  25.676278
3  H800   1        32  0.157788  0.126257   0.207788  0.092353      0.045651      0.562852  0.051176  0.816290  0.143529 -0.109909  48.196983  48.196983  20.748187
4  H800   4        32  0.157788  0.126257   0.123617  0.092353      0.045651      0.562852  0.051176  0.732120  0.143529 -0.025739  43.315086  43.315086  23.086645
5  H800   1        64  0.263094  0.141337   0.313094  0.174706      0.058217      0.663386  0.092353  1.034698  0.267059 -0.104253  61.225801  61.225801  16.332983
    """
    df['TPS'] = 1000 / df['TPOT']
    # 没有调整过的TPS
    df['TPS_O'] = 1000 / df['TPOT_O']
    # 所有batch 的TPS 总和：1个卡的TPS------之前计算的仅仅是1个batch的数据吗？？？
    df['Total'] = df['TPS'] * df['BatchSize'].astype(int)
    #_O是所有batch的，做了overlap调整后的 TPS，就是吞吐
    """
    GPU  TP BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS      TPS_O        Total      Total_O
0  H800   1        16  0.105135  0.118717   0.155135  0.051176      0.041462      0.529340  0.030588  0.725937  0.081765 -0.114832  42.775888  42.775888  23.377656  23.377656   374.042497   374.042497
1  H800   4        16  0.105135  0.118717   0.100869  0.051176      0.041462      0.529340  0.030588  0.671671  0.081765 -0.060566  39.628464  39.628464  25.234387  25.234387   403.750188   403.750188
2  H800   8        16  0.105135  0.118717   0.089110  0.051176      0.041462      0.529340  0.030588  0.659912  0.081765 -0.048807  38.946455  38.946455  25.676278  25.676278   410.820444   410.820444
3  H800   1        32  0.157788  0.126257   0.207788  0.092353      0.045651      0.562852  0.051176  0.816290  0.143529 -0.109909  48.196983  48.196983  20.748187  20.748187   663.941972   663.941972
4  H800   4        32  0.157788  0.126257   0.123617  0.092353      0.045651      0.562852  0.051176  0.732120  0.143529 -0.025739  43.315086  43.315086  23.086645  23.086645   738.772625   738.772625
5  H800   1        64  0.263094  0.141337   0.313094  0.174706      0.058217      0.663386  0.092353  1.034698  0.267059 -0.104253  61.225801  61.225801  16.332983  16.332983  1045.310939  1045.310939
    H800卡，16张卡，TP=1，batch size=32，1张卡吞吐738tokens/s
    不是全部16张卡，参考
    https://mp.weixin.qq.com/s/214lYyKmL3XmPUHnnTrbXg?poc_token=HEihT2qjOmuwBZjM7iz4mO4EeIbr51leI-lWXhfX
    满足用户TPS>10,则H800需要BatchSize<=128, 此时H800峰值每卡每秒可以产生3984个token, 
    """
    df['Total_O'] = df['TPS_O'] * df['BatchSize'].astype(int)
    #bp()
    # 1M 吞吐/美元，tokens/sec/dollar, 含义：输出几M tokens/$，每美元成本的吞吐
    df['Total_per_dollar'] = df['Total']*3600/(1000*1000) /  df['HourlyCost']
    df['Total_O_per_dollar'] = df['Total_O']*3600/(1000*1000) /  df['HourlyCost']

    df['Comm_Impact'] = (df['Total_O'] - df['Total']) / df['Total_O']

    df = df[df['TPS'] > tps_limit]
    if print_console:
        print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))
    return df

"""
得到不同EP（卡数）的decode时间
比如
(Pdb) dd
     GPU TP  EP  BatchSize  DenseMLA  DenseMLP  SparseMLA   Combine  SharedExpert  RoutedExpert  Dispatch  COMP_SUM  COMM_SUM     Delta       TPOT     TPOT_O        TPS      TPS_O        Total      Total_O  Comm_Impact
0   H800  1  16         16  0.105135  0.118717   0.155135  0.051176      0.041462      0.529340  0.030588  0.725937  0.081765 -0.114832  42.775888  42.775888  23.377656  23.377656   374.042497   374.042497          0.0
1   H800  4  16         16  0.105135  0.118717   0.100869  0.051176      0.041462      0.529340  0.030588  0.671671  0.081765 -0.060566  39.628464  39.628464  25.234387  25.234387   403.750188   403.750188          0.0
2   H800  8  16         16  0.105135  0.118717   0.089110  0.051176      0.041462      0.529340  0.030588  0.659912  0.081765 -0.048807  38.946455  38.946455  25.676278  25.676278   410.820444   410.820444          0.0
3   H800  1  16         32  0.157788  0.126257   0.207788  0.092353      0.045651      0.562852  0.051176  0.816290  0.143529 -0.109909  48.196983  48.196983  20.748187  20.748187   663.941972   663.941972          0.0
4   H800  4  16         32  0.157788  0.126257   0.123617  0.092353      0.045651      0.562852  0.051176  0.732120  0.143529 -0.025739  43.315086  43.315086  23.086645  23.086645   738.772625   738.772625          0.0
5   H800  1  16         64  0.263094  0.141337   0.313094  0.174706      0.058217      0.663386  0.092353  1.034698  0.267059 -0.104253  61.225801  61.225801  16.332983  16.332983  1045.310939  1045.310939          0.0
6   H800  1  36         16  0.105135  0.118717   0.155135  0.051176      0.035178      0.281426  0.030588  0.471739  0.081765 -0.108548  28.032416  28.032416  35.672986  35.672986   570.767782   570.767782          0.0
7   H800  4  36         16  0.105135  0.118717   0.100869  0.051176      0.035178      0.281426  0.030588  0.417473  0.081765 -0.054282  24.884993  24.884993  40.184862  40.184862   642.957797   642.957797          0.0
8   H800  8  36         16  0.105135  0.118717   0.089110  0.051176      0.035178      0.281426  0.030588  0.405714  0.081765 -0.042524  24.202983  24.202983  41.317221  41.317221   661.075531   661.075531          0.0
9   H800  1  36         32  0.157788  0.126257   0.207788  0.092353      0.041462      0.331693  0.051176  0.580943  0.143529 -0.105720  34.546825  34.546825  28.946220  28.946220   926.279034   926.279034          0.0
10  H800  4  36         32  0.157788  0.126257   0.123617  0.092353      0.041462      0.331693  0.051176  0.496772  0.143529 -0.021550  29.664929  29.664929  33.709840  33.709840  1078.714886  1078.714886          0.0
11  H800  1  36         64  0.263094  0.141337   0.313094  0.174706      0.041462      0.331693  0.092353  0.686249  0.267059 -0.087497  41.015759  41.015759  24.380873  24.380873  1560.375865  1560.375865          0.0
12  H800  1  36        128  0.473707  0.171497   0.523707  0.339412      0.043855      0.350843  0.174706  0.918405  0.514118 -0.053445  55.203127  55.203127  18.114916  18.114916  2318.709212  2318.709212          0.0
22  H800  1  144         16  0.105135  0.118717   0.155135  0.051176      0.028895      0.082923  0.030588  0.266953  0.081765 -0.102265  16.154827  16.154827   61.901005   61.901005   990.416076   990.416076     0.000000
23  H800  4  144         16  0.105135  0.118717   0.100869  0.051176      0.028895      0.082923  0.030588  0.212687  0.081765 -0.047999  13.007403  13.007403   76.879297   76.879297  1230.068751  1230.068751     0.000000
24  H800  8  144         16  0.105135  0.118717   0.089110  0.051176      0.028895      0.082923  0.030588  0.200928  0.081765 -0.036240  12.325394  12.325394   81.133311   81.133311  1298.132975  1298.132975     0.000000
25  H800  1  144         32  0.157788  0.126257   0.207788  0.092353      0.029493      0.087711  0.051176  0.324992  0.143529 -0.093752  19.701669  19.701669   50.757122   50.757122  1624.227893  1624.227893     0.000000
26  H800  4  144         32  0.157788  0.126257   0.123617  0.092353      0.029493      0.087711  0.051176  0.240821  0.143529 -0.009581  14.819772  14.819772   67.477422   67.477422  2159.277489  2159.277489     0.000000
27  H800  8  144         32  0.157788  0.126257   0.106660  0.092353      0.029493      0.087711  0.051176  0.223864  0.143529  0.007376  14.264059  13.836245   70.106271   72.273944  2243.400686  2312.766209     0.029992
28  H800  1  144         64  0.263094  0.141337   0.313094  0.174706      0.033084      0.116435  0.092353  0.462613  0.267059 -0.079119  28.044855  28.044855   35.657164   35.657164  2282.058508  2282.058508     0.000000
29  H800  4  144         64  0.263094  0.141337   0.169114  0.174706      0.033084      0.116435  0.092353  0.318633  0.267059  0.064861  23.455927  19.694012   42.633147   50.776856  2728.521402  3249.718791     0.160382
30  H800  1  144        128  0.473707  0.171497   0.523707  0.339412      0.035876      0.138776  0.174706  0.698360  0.514118 -0.045466  42.440474  42.440474   23.562414   23.562414  3015.988936  3015.988936     0.000000
31  H800  1  144        256  0.894933  0.231818   0.944933  0.668824      0.043855      0.202607  0.339412  1.191396  1.008235  0.019447  73.609130  72.481214   13.585271   13.796678  3477.829461  3531.949680     0.015323

EP144卡，batch=128，单卡吞吐3015===OK
(Pdb) 
"""
def decode_time_with_ep_list(args: ModelArgs, gpu_dict,
                             config: Config,
                             tps_limit=0,
                             fp8_combine=False,
                             print_console=False,
                             mem_read_latency=False):
    df_list = []
    #bp()
    # for device_num in [8, 16, 36, 72, 144, 320],GPU卡数, 为什么卡数=8时，没有返回结果？？？
    for device_num in config.eplist:
        # 每个设备上的gemm：总路由专家数目256/GPU卡数
        gemm_group_per_device = math.ceil(args.n_routed_experts / device_num)
        # 对每个GPU卡数，计算解码时间，返回一个df
        df = decode_time(args, gpu_dict, config.bs_list, config.seq_len,
                         config.decode_len,
                         gemm_group_per_device=gemm_group_per_device,
                         device_num=device_num,
                         fp8_combine=fp8_combine,
                         tps_limit=tps_limit,
                         print_console=False,
                         mem_read_latency=mem_read_latency)
        df['EP'] = device_num
        df_list.append(df)
        #bp()
    dd = pd.concat(df_list)
    dd.reset_index(inplace=True, drop=True)
    order = ['GPU', 'TP', 'EP', 'BatchSize', 'DenseMLA', 'DenseMLP', 'SparseMLA',
             'Combine', 'SharedExpert', 'RoutedExpert', 'Dispatch', 'COMP_SUM',
             'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O', 'TPS', 'TPS_O', 'Total',
             'Total_O', 'Comm_Impact','Total_per_dollar','Total_O_per_dollar']
    dd = dd[order]
    dd['BatchSize'] = dd['BatchSize'].astype(int)
    #bp()
    return dd


def df_filter(df,gpu,device_num=0, bs=0,tps_limit=0, value_list=[]):
    df_o = df[df['GPU'] == gpu] 
    if bs > 0:
        df_o = df_o[df_o['BatchSize'] == str(bs)]
    if tps_limit > 0:
        df_o = df_o[df_o['TPS'] > tps_limit]
    if device_num > 0:
        df_o = df_o[df_o['EP'] == device_num]
    if len(value_list) > 0:
        df_o = df_o[value_list]
    return df_o



def df_sort(df,value,ascending=False):
    if ascending:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].min()]).sort_values([value],ascending=True).reset_index(drop=True)
    else:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].max()]).sort_values([value],ascending=False).reset_index(drop=True)
    return df_o

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def color_positive_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for positive
    strings, black otherwise.
    """
    color = 'red' if val > 0 else 'black'
    return 'color: %s' % color

def gpu_category_color(s,props):
    colors = ['color:darkred','color:steelblue','color:green','color:black','color:m',
              'color:darkgoldenrod','color:darkgreen','color:crimson','color:brwon','color:sienna',
              'color:navy','color:pink','color:gray','color:darkviolet']
    gpu_idx = props[s]
    return colors[gpu_idx]

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)



def draw(df, gpu_dict,
         comp_name,comp_val_list, 
         val_list, val_unit_name,
         title, width=8, height=4,
         filename='',savefig=False):
    def _df_filter(df, gpu, comp_name, comp_val, value_list):
        df1 = df[(df['GPU'] == gpu) & (df[comp_name] == comp_val)][value_list]
        return df1
    
    sns.color_palette("Paired")
    num_gpu = len(gpu_dict)
    fig_height = height * num_gpu
    fig_width = width * len(val_list)
    fig, axs = plt.subplots(nrows=num_gpu, ncols=len(val_list), figsize=(fig_width, fig_height))

    fig.suptitle(title, y=0.9,fontsize='large')
    value_list = val_list + ['index_value']
    cnt = 0
    for key in gpu_dict.keys():
        axt = axs[cnt]
        for i in range(0,len(val_list)):
            axt[i].legend(comp_val_list)
        for comp_v in comp_val_list:
            df1 = _df_filter(df, key , comp_name, comp_v, value_list)
            for i in range(0,len(val_list)):
                sns.lineplot(x='index_value', y=val_list[i],label=str(comp_v), data=df1,  ax=axt[i])
                axt[i].set_ylabel(val_unit_name)
                axt[i].set_xlabel(key+'('+val_list[i]+')')
        cnt += 1

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.2, hspace=0.2)
    if savefig:
        if filename=="":
            filename = './figures/'+title.replace(' ','_')+'.png'
        plt.savefig(filename,bbox_inches='tight', pad_inches=0.05)
    plt.show()


def generate_gpu_comparison_html(df, ep_list = [1, 2, 4, 8, 16],gpu_map = {"DGX-B200": "B200",  "B200-HBF-lowLatency": "B200-HBF"},output_file="gpu_comparison.html"):
    # 1. 规范化 GPU 名称映射（兼容 DGX-B200 与 B200）
    df_copy = df.copy()
    #gpu_map = {"DGX-B200": "B200",  "B200-HBF-lowLatency": "B200-HBF"}
    df_copy["GPU_Group"] = df_copy["GPU"].map(gpu_map)

    # 目标 EP 列表
    ep_list = ep_list

    # 2. 查找每个 GPU 组在不同 EP 下 Total 最大的行
    filtered = df_copy[
        df_copy["GPU_Group"].isin(["B200", "B200-HBF"])
        & df_copy["EP"].isin(ep_list)
    ]

    # 按 GPU_Group 和 EP 分组获取 Total 最大值所在位置
    idx = filtered.groupby(["GPU_Group", "EP"])["Total"].idxmax()
    max_df = filtered.loc[idx]

    # 3. 提取数据填充数据表字典
    data = {
        gpu: {
            "throughput": {ep: np.nan for ep in ep_list},
            "tpd": {ep: np.nan for ep in ep_list},
        }
        for gpu in ["B200", "B200-HBF"]
    }

    for _, row in max_df.iterrows():
        gpu = row["GPU_Group"]
        ep = int(row["EP"])
        data[gpu]["throughput"][ep] = row["Total"]
        data[gpu]["tpd"][ep] = row["Total_per_dollar"]
    #bp()
    # 4. 构建输出表格结构
    rows = []

    # --- B200 行 ---
    b200_tp = [
        (
            f"{int(round(data['B200']['throughput'][ep]))}"
            if not np.isnan(data["B200"]["throughput"][ep])
            else "-"
        )
        for ep in ep_list
    ]
    b200_tpd = [
        (
            f"{data['B200']['tpd'][ep]:.1f}"
            if not np.isnan(data["B200"]["tpd"][ep])
            else "-"
        )
        for ep in ep_list
    ]

    rows.append(
        ["B200", "throughput<br>tokens/s"] + b200_tp,
    )
    rows.append(
        ["", "TPD每美元吞吐<br>M token/$"] + b200_tpd,
    )

    # --- B200-HBF 行 ---
    hbf_tp = [
        (
            f"{int(round(data['B200-HBF']['throughput'][ep]))}"
            if not np.isnan(data["B200-HBF"]["throughput"][ep])
            else "-"
        )
        for ep in ep_list
    ]
    hbf_tpd = [
        (
            f"{data['B200-HBF']['tpd'][ep]:.1f}"
            if not np.isnan(data["B200-HBF"]["tpd"][ep])
            else "-"
        )
        for ep in ep_list
    ]

    # --- 倍数计算 ---
    tp_mult = []
    tpd_mult = []
    for ep in ep_list:
        b_tp = data["B200"]["throughput"][ep]
        h_tp = data["B200-HBF"]["throughput"][ep]
        b_tpd = data["B200"]["tpd"][ep]
        h_tpd = data["B200-HBF"]["tpd"][ep]

        # 计算 throughput 倍数
        if not np.isnan(b_tp) and not np.isnan(h_tp) and b_tp > 0:
            tp_mult.append(f"{h_tp / b_tp:.1f}")
        else:
            tp_mult.append("-")

        # 计算 TPD 倍数
        if not np.isnan(b_tpd) and not np.isnan(h_tpd) and b_tpd > 0:
            tpd_mult.append(f"{h_tpd / b_tpd:.2f}".rstrip("0").rstrip("."))
        else:
            tpd_mult.append("-")

    rows.append(
        ["B200-HBF", "throughput"] + hbf_tp,
    )
    rows.append(
        ["", "M token/$"] + hbf_tpd,
    )
    rows.append(
        ["", "throughput X倍数"] + tp_mult,
    )
    rows.append(
        ["", "TPD X"] + tpd_mult,
    )

    # 5. 生成 HTML 代码段（含 CSS 样式）
    html_content_1 = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
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
    """
    html_content_2 = f"""
    <table>
        <thead>
            <tr>
                <th colspan="2">GPU \\ EP</th>
                {"".join([f"<th>{ep}</th>" for ep in ep_list])}
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2" class="gpu-header">B200</td>
                <td>{rows[0][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[0][2:]])}
            </tr>
            <tr>
                <td>{rows[1][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[1][2:]])}
            </tr>
            <tr>
                <td rowspan="4" class="gpu-header">B200-HBF</td>
                <td>{rows[2][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[2][2:]])}
            </tr>
            <tr>
                <td>{rows[3][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[3][2:]])}
            </tr>
            <tr>
                <td>{rows[4][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[4][2:]])}
            </tr>
            <tr>
                <td>{rows[5][1]}</td>
                {"".join([f"<td>{val}</td>" for val in rows[5][2:]])}
            </tr>
        </tbody>
    </table>
    """
    html_content_3 = f"""
    </body>
    </html>
    """

    # 保存到 HTML 文件
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML 文件已成功保存至: {output_file}")
    """
    return html_content_2