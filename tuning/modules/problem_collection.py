""" Functions of problem collection """
import torch
import yaml
import os
from typing import Dict
from flash_attn.tuning.modules.parse_problem_list import parse_origin_to_problem

def problem_collection(func_name, is_bf16, data: Dict):
    origin_input = {}
    for key, value in data.items():
        if key == "cu_seqlens_q" or key == "cu_seqlens_k":
            origin_input[key] = value.cpu().tolist()
        elif torch.is_tensor(value):
            if key in ["window_size_left","window_size_right","max_seqlen_q","max_seqlen_k"] :
                origin_input[key] = value.item()
            else:
                origin_input[key] = list(value.shape)
        else:
            origin_input[key] = value

    origin_input["func_name"] = func_name
    if is_bf16:
        origin_input["dtype"] = "bf16"
    else:
        origin_input["dtype"] = "fp16"

    # parse to problem
    prob = parse_origin_to_problem(origin_input)

    return prob
