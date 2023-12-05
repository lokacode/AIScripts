# script to convert fp32 model to fp16
# Usage: python fp32_fp16.py <fp32 model folder> <fp16 output folder> --device <device>
# Example: python fp32_fp16.py gpt2 gpt2_fp16 --device cuda:0

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

parser = argparse.ArgumentParser(description='Convert fp32 model to fp16')
parser.add_argument('model_dir', type=str, help='fp32 model folder')
parser.add_argument('output_dir', type=str, help='fp16 output folder')
parser.add_argument('--device', type=str, default="cuda:0", help='device')

args = parser.parse_args()

model_dir =  args.model_dir
output_dir = args.output_dir

model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            )

model = model.half()

model.save_pretrained(
            output_dir, torch_dtype=torch.float16
            )