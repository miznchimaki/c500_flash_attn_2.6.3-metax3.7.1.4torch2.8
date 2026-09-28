import flash_attn_2_cuda as flash_attn_cuda
import yaml
import torch
import argparse
import glob
import os
from tqdm import tqdm
import sys
from flash_attn.tuning.modules.logger import logger

flash_attn_fwd          = "_flash_attn_forward"
flash_attn_varlen_fwd   = "_flash_attn_varlen_forward"
flash_attn_bwd          = "_flash_attn_backward"
flash_attn_varlen_bwd   = "_flash_attn_varlen_backward"
flash_attn_fwd_kvcache  = "flash_attn_with_kvcache"

flash_api_dict = {
    flash_attn_fwd : flash_attn_cuda.FLASH_FWD,
    flash_attn_varlen_fwd : flash_attn_cuda.FLASH_VARLEN_FWD,
    flash_attn_bwd : flash_attn_cuda.FLASH_BWD,
    flash_attn_varlen_bwd : flash_attn_cuda.FLASH_VARLEN_BWD,
    flash_attn_fwd_kvcache : flash_attn_cuda.FLASH_FWD_KVCACHE
}

def problem_to_dict(prob):

    varlen_api = [flash_attn_cuda.FLASH_VARLEN_FWD, flash_attn_cuda.FLASH_VARLEN_BWD]
    if prob.api in varlen_api:
        seqlens_q = [prob.seqlens_q[i+1] - prob.seqlens_q[i] for i in range(len(prob.seqlens_q) - 1)]
        seqlens_kv = [prob.seqlens_kv[i+1] - prob.seqlens_kv[i] for i in range(len(prob.seqlens_kv) - 1)]
    else:
        seqlens_q = prob.seqlens_q
        seqlens_kv = prob.seqlens_kv

    api_to_string = dict(zip(flash_api_dict.values(), flash_api_dict.keys()))
    api_string = api_to_string[prob.api]

    prob_dict = {
        "head_dim" : prob.head_dim,
        "head_dim_v" : prob.head_dim_v,
        "num_heads_q" : prob.num_heads_q,
        "num_heads_kv" : prob.num_heads_kv,
        "batch_size" : prob.batch_size,
        "batch_size_c": prob.batch_size_c,
        'max_seqlen_q': prob.max_seqlen_q,
        'max_seqlen_kv': prob.max_seqlen_kv,
        'total_seqlens_q': prob.total_seqlens_q,
        'total_seqlens_kv': prob.total_seqlens_kv,
        "seqlens_q": seqlens_q,
        "seqlens_kv": seqlens_kv,
        'causal': prob.causal,
        'dropout': prob.dropout,
        'alibi': prob.alibi,
        'attn_mask': prob.attn_mask,
        'deterministic': prob.deterministic,
        'qkvpacked': prob.qkvpacked,
        'kvpacked': prob.kvpacked,
        'window_left': prob.window_left,
        'window_right': prob.window_right,
        'paged_kv': prob.paged_kv,
        'paged_block_size': prob.paged_block_size,
        'paged_num_blocks': prob.paged_num_blocks,
        'append_kv': prob.append_kv,
        'rope': prob.rope,
        'softcap': prob.softcap,
        "is_fp16": prob.is_fp16,
        'api': api_string,
        'hash_code': prob.hash_code,
    }

    if prob.api in varlen_api:
        prob_dict["cu_seqlens_q"] = prob.seqlens_q
        prob_dict["cu_seqlens_kv"] = prob.seqlens_kv

    return prob_dict

def parse_fwd(prob_dict):
    prob = flash_attn_cuda.Flash_problem()
    prob.batch_size = prob_dict["q"][0]
    prob.num_heads_q = prob_dict["q"][2]
    prob.head_dim = prob_dict["q"][3]

    prob.num_heads_kv = prob_dict["k"][2]
    prob.head_dim_v = prob_dict["v"][3]

    prob.seqlens_q = [prob_dict["q"][1]] * prob.batch_size
    prob.seqlens_kv = [prob_dict["k"][1]] * prob.batch_size

    prob.max_seqlen_q = prob_dict["q"][1]
    prob.max_seqlen_kv = prob_dict["k"][1]
    prob.total_seqlens_q = prob.max_seqlen_q * prob.batch_size
    prob.total_seqlens_kv = prob.max_seqlen_kv * prob.batch_size

    prob.softcap = prob_dict.get("softcap",0.0)

    prob.causal = prob_dict["causal"]
    prob.dropout = prob_dict["dropout_p"] != 0
    prob.alibi = prob_dict["alibi_slopes"] is not None
    prob.attn_mask = prob_dict["attn_mask"] is not None
    prob.qkvpacked = prob_dict["qkvpacked"] if "qkvpacked" in prob_dict else False
    prob.kvpacked = prob_dict["kvpacked"] if "kvpacked" in prob_dict else False

    prob.window_left = prob_dict["window_size_left"]
    prob.window_right = prob_dict["window_size_right"]

    prob.is_fp16 = prob_dict["dtype"] == "fp16"
    prob.api = flash_api_dict[flash_attn_fwd]

    prob.hash_code = flash_attn_cuda.flash_problem_hash(prob)

    return prob

def parse_bwd(prob_dict):

    prob = flash_attn_cuda.Flash_problem()
    prob.batch_size = prob_dict["q"][0]
    prob.num_heads_q = prob_dict["q"][2]
    prob.head_dim = prob_dict["q"][3]

    prob.num_heads_kv = prob_dict["k"][2]
    prob.head_dim_v = prob_dict["v"][3]

    prob.seqlens_q = [prob_dict["q"][1]] * prob.batch_size
    prob.seqlens_kv = [prob_dict["k"][1]] * prob.batch_size

    prob.max_seqlen_q = prob_dict["q"][1]
    prob.max_seqlen_kv = prob_dict["k"][1]
    prob.total_seqlens_q = prob.max_seqlen_q * prob.batch_size
    prob.total_seqlens_kv = prob.max_seqlen_kv * prob.batch_size

    prob.causal = prob_dict["causal"]
    prob.dropout = prob_dict["dropout_p"] != 0
    prob.alibi = prob_dict["alibi_slopes"] is not None
    prob.attn_mask = prob_dict["attn_mask"] is not None
    prob.deterministic = prob_dict["deterministic"]
    prob.qkvpacked = prob_dict["qkvpacked"] if "qkvpacked" in prob_dict else False
    prob.kvpacked = prob_dict["kvpacked"] if "kvpacked" in prob_dict else False

    prob.window_left = prob_dict["window_size_left"]
    prob.window_right = prob_dict["window_size_right"]

    prob.softcap = prob_dict.get("softcap",0.0)

    prob.is_fp16 = prob_dict["dtype"] == "fp16"
    prob.api = flash_api_dict[flash_attn_bwd]

    prob.hash_code = flash_attn_cuda.flash_problem_hash(prob)

    return prob

def parse_varlen_fwd(prob_dict):

    prob = flash_attn_cuda.Flash_problem()
    prob.batch_size = len(prob_dict["cu_seqlens_q"]) - 1
    paged_kv = prob_dict.get("block_table") is not None
    prob.paged_kv = paged_kv

    if paged_kv:
        prob.paged_num_blocks = prob_dict["k"][0]
        prob.paged_block_size = prob_dict["k"][1]
        prob.num_heads_kv = prob_dict["k"][2]
        prob.head_dim_v = prob_dict["v"][3]
        max_num_blocks_per_seq = prob_dict["block_table"][1]
        prob.batch_size_c = prob.batch_size
        prob.total_seqlens_kv = -1
    else:
        prob.total_seqlens_kv = prob_dict["k"][0]
        prob.num_heads_kv = prob_dict["k"][1]
        prob.head_dim_v = prob_dict["v"][2]

    prob.num_heads_q = prob_dict["q"][1]
    prob.head_dim = prob_dict["q"][2]


    prob.seqlens_q = prob_dict["cu_seqlens_q"]
    prob.seqlens_kv = prob_dict["cu_seqlens_k"]

    prob.max_seqlen_q = prob_dict["max_seqlen_q"]
    prob.max_seqlen_kv = prob_dict["max_seqlen_k"]
    prob.total_seqlens_q = prob_dict["q"][0]

    prob.causal = prob_dict["causal"]
    prob.dropout = prob_dict["dropout_p"] != 0
    prob.alibi = prob_dict["alibi_slopes"] is not None
    prob.qkvpacked = prob_dict["qkvpacked"] if "qkvpacked" in prob_dict else False
    prob.kvpacked = prob_dict["kvpacked"] if "kvpacked" in prob_dict else False

    prob.window_left = prob_dict["window_size_left"]
    prob.window_right = prob_dict["window_size_right"]

    prob.softcap = prob_dict.get("softcap",0.0)

    prob.is_fp16 = prob_dict["dtype"] == "fp16"
    prob.api = flash_api_dict[flash_attn_varlen_fwd]

    prob.hash_code = flash_attn_cuda.flash_problem_hash(prob)

    return prob

def parse_varlen_bwd(prob_dict):

    prob = flash_attn_cuda.Flash_problem()
    prob.batch_size = len(prob_dict["cu_seqlens_q"]) - 1

    prob.num_heads_q = prob_dict["q"][1]
    prob.head_dim = prob_dict["q"][2]

    prob.num_heads_kv = prob_dict["k"][1]
    prob.head_dim_v = prob_dict["v"][2]

    prob.seqlens_q = prob_dict["cu_seqlens_q"]
    prob.seqlens_kv = prob_dict["cu_seqlens_k"]

    prob.max_seqlen_q = prob_dict["max_seqlen_q"]
    prob.max_seqlen_kv = prob_dict["max_seqlen_k"]
    prob.total_seqlens_q = prob_dict["q"][0]
    prob.total_seqlens_kv = prob_dict["k"][0]

    prob.causal = prob_dict["causal"]
    prob.dropout = prob_dict["dropout_p"] != 0
    prob.alibi = prob_dict["alibi_slopes"] is not None
    prob.deterministic = prob_dict["deterministic"]
    prob.qkvpacked = prob_dict["qkvpacked"] if "qkvpacked" in prob_dict else False
    prob.kvpacked = prob_dict["kvpacked"] if "kvpacked" in prob_dict else False

    prob.window_left = prob_dict["window_size_left"]
    prob.window_right = prob_dict["window_size_right"]

    prob.softcap = prob_dict.get("softcap",0.0)

    prob.is_fp16 = prob_dict["dtype"] == "fp16"
    prob.api = flash_api_dict[flash_attn_varlen_bwd]

    prob.hash_code = flash_attn_cuda.flash_problem_hash(prob)

    return prob

def parse_fwd_kvcache(prob_dict):

    prob = flash_attn_cuda.Flash_problem()

    prob.batch_size = prob_dict["q"][0]
    prob.num_heads_q = prob_dict["q"][2]
    prob.head_dim = prob_dict["q"][3]
    prob.max_seqlen_q = prob_dict["q"][1]
    prob.seqlens_q = [prob_dict["q"][1]] * prob.batch_size

    prob.num_heads_kv = prob_dict["k_cache"][2]
    prob.head_dim_v = prob_dict["v_cache"][3]

    paged_kv = prob_dict["block_table"] is not None
    prob.paged_kv = paged_kv

    if paged_kv:
        prob.paged_num_blocks = prob_dict["k_cache"][0]
        prob.paged_block_size = prob_dict["k_cache"][1]

        max_num_blocks_per_seq = prob_dict["block_table"][1]
        prob.max_seqlen_kv = max_num_blocks_per_seq * prob.paged_block_size
        prob.batch_size_c = prob.batch_size
    else:
        prob.batch_size_c = prob_dict["k_cache"][0]
        prob.max_seqlen_kv = prob_dict["k_cache"][1]
        prob.seqlens_kv = [prob_dict["k_cache"][1]] * prob.batch_size_c

    prob.causal = prob_dict["causal"]
    prob.alibi = prob_dict["alibi_slopes"] is not None

    prob.window_left = prob_dict["window_size_left"]
    prob.window_right = prob_dict["window_size_right"]

    prob.softcap = prob_dict.get("softcap",0.0)

    prob.is_fp16 = prob_dict["dtype"] == "fp16"
    prob.api = flash_api_dict[flash_attn_fwd_kvcache]
    prob.hash_code = flash_attn_cuda.flash_problem_hash(prob)

    return prob

def parse_origin_to_problem(origin_input):
    flash_problem_parser = {
        flash_attn_fwd : parse_fwd,
        flash_attn_varlen_fwd : parse_varlen_fwd,
        flash_attn_bwd : parse_bwd,
        flash_attn_varlen_bwd : parse_varlen_bwd,
        flash_attn_fwd_kvcache : parse_fwd_kvcache
    }

    func_name = origin_input["func_name"]

    prob = flash_problem_parser[func_name](origin_input)

    return problem_to_dict(prob)

def parse_probelm(input_dir, output_file):

    flash_problem_parser = {
        flash_attn_fwd : parse_fwd,
        flash_attn_varlen_fwd : parse_varlen_fwd,
        flash_attn_bwd : parse_bwd,
        flash_attn_varlen_bwd : parse_varlen_bwd,
        flash_attn_fwd_kvcache : parse_fwd_kvcache
    }

    problem_count = {}
    input_list = []
    pattern = os.path.join(input_dir, "problem_list_*.yaml")
    input_flie_list = glob.glob(pattern)
    if len(input_flie_list) == 0:
        logger.warning(f"Found no problem_list yaml file form {input_dir}")
        return False

    logger.info(f"Start parse problem...")
    logger.info(f"input dir:{input_dir}")

    for input_file in tqdm(input_flie_list, desc='Processing', ascii=False, unit='task'):
        logger.info(f"parse file : {input_file}")
        with open(input_file, "r") as file:
            problems = yaml.load(file, Loader=yaml.FullLoader)

        for name in problems:
            origin_prob_dict = problems[name]
            func_name = origin_prob_dict["func_name"]

            prob = flash_problem_parser[func_name](origin_prob_dict)
            if prob.hash_code in problem_count:
                problem_count[prob.hash_code] += 1
            else:
                problem_count[prob.hash_code] = 1

                prob_dict = problem_to_dict(prob)
                origin_prob_dict["seqlens_q"] = prob_dict["seqlens_q"]
                origin_prob_dict["seqlens_kv"] = prob_dict["seqlens_kv"]

                input_packed = {}
                input_packed["problem"] = prob_dict
                input_packed["origin"] = origin_prob_dict

                input_list.append(input_packed)

    probs = []
    for index, prob_packed in enumerate(input_list):
        hash_code = prob_packed["problem"]["hash_code"]
        dict = {
            "name": f"input_{index}",
            "count":problem_count[hash_code],
            "origin":prob_packed["origin"],
            "prob":prob_packed["problem"]
        }
        probs.append(dict)

    with open(output_file, 'w') as file:
        yaml.dump(probs, file)

    logger.info(f"Parse problem list completed,output file:{output_file}")
    return True