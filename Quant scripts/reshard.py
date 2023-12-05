# script to reshard a model
# Usage: python reshard.py --base_model_name_or_path <model folder> --output_dir <output folder> --max_shard_size <max shard size> --dtype <float16|bfloat16|float32> --device <device> --trust_remote_code
# Example: python reshard.py --base_model_name_or_path "/content/Mytho" --output_dir "/content/Mytho-reshard" --max_shard_size "9GiB" --dtype "float16" --device "cuda:0" --trust_remote_code

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_shard_size", type=str, default="9GiB")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    if args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.dtype == 'float32':
        dtype = torch.bfloat32

    print(f"Loading base model: {args.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        **device_arg
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir} with max_shard_size={args.max_shard_size}")

if __name__ == "__main__" :
    main()