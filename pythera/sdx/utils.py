import torch

def get_dtype_training(dtype):
    if dtype == 'bf16':
        return torch.bfloat16
    if dtype == 'fp16':
        return torch.float16
    else:
        return torch.float32
    
