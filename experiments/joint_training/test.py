import torch, csv
import torch.utils.benchmark as benchmark
import torch
import torch.nn as nn
from torch.profiler import *

from jointformer.models.layers.attention import Attention
from jointformer.models.layers.gqa import GroupedQueryAttention


def init_csv_writer(filename):
    header = ['algorithm', 'batch_size', 'sequence_length', 'embed_dim', 'num_heads', 'group_size', 'execution_time']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


def write_to_csv(filename, algorithm, batch_size, seq_len, embed_dim, num_heads, group_size, exec_time):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([algorithm, batch_size, seq_len, embed_dim, num_heads, group_size, exec_time])


def create_attn(batch_size, seq_len):
    embed_dim = 512
    max_seq_len = 1024
    num_heads = 8
    group_size = 2
    att = Attention(embedding_dim=embed_dim, num_heads=num_heads, bias=False, dropout=0, block_size=max_seq_len).to("cuda")
    gqa = GroupedQueryAttention(embedding_dim=embed_dim, num_q_heads=num_heads, group_size=group_size, bias=False, dropout=0, max_seq_len=max_seq_len).to("cuda")
    gqa.update_training_mode(True)
    rand = torch.randn((batch_size, seq_len, embed_dim), device="cuda")
    return att, gqa, rand

    
def timed(fn, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)
    

def prof(name, mechanism, input_data, data):
    data[name]
    for _ in range(100):
        timed(mechanism.forward, input_data, None, True)
    for _ in range(10000):
        _, time = timed(mechanism.forward, input_data, None, True)
    print(f"{name}: {time:.3f}s")
    

data = {}

filename = 'attention_results.csv'
init_csv_writer(filename)
for bs in (128, 256, 512):
    for seq_len in (128, 256, 512, 1024):
        att, gqa, rand = create_attn(bs, seq_len)
        att.flash = False
        prof("Self-Attention (Native)", att, rand, data)
        att.flash = True
        prof("Self-Attention (Optimized)", att, rand, data)
        prof("Grouped-Query Attention (Native)", gqa, rand, data)






"""
Do CUDA stuff here

Do KV-Cache as well (use small smiles from guacamol)

execution will be measured like this
novelty etc is done
comparing increasing batch_sizes

on friday, code molgpt stuff and get values to include into the thesis
"""