""" Functions of problem collection """
import torch
import yaml
import os

env_input_collection = "MHA_INPUT_COLLECTION"

class count_calls(object):
    def __init__ (self, func):
        self.func = func
        self.call_count = 0

    def __call__ (self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, call_count = self.call_count, **kwargs)

@count_calls
def problem_collection(func_name,is_bf16,kwargs,call_count=0):
    problem = {}
    for key, value in kwargs.items():
        if key == "cu_seqlens_q" or key == "cu_seqlens_k":
            problem[key] = value.cpu().tolist()
        elif torch.is_tensor(value):
            if key in ["window_size_left","window_size_right"] :
                problem[key] = value.item()
            else:
                problem[key] = list(value.shape)
        else:
            problem[key] = value

    problem["func_name"] = func_name
    if is_bf16:
        problem["dtype"] = "bf16"
    else:
        problem["dtype"] = "fp16"

    output_dir = os.getenv(env_input_collection)
    os.makedirs(output_dir, exist_ok=True)

    pid = os.getpid()
    file_name = f"{output_dir}/problem_list_{pid}.yaml"

    with open(file_name, 'a') as file:
        yaml.dump({
            "problem_{}".format(call_count): problem
        }, file)


def is_qkvpacked(q, is_varlen = False):
    seqlen = 1 if is_varlen else q.shape[1]
    heads = q.shape[1] if is_varlen else q.shape[2]
    head_size = q.shape[2] if is_varlen else q.shape[3]

    batch_stride = q.stride()[0]
    nopacked_stride = seqlen * heads * head_size

    return batch_stride == 3 * nopacked_stride

def is_kvpacked(k, is_varlen = False):
    seqlen = 1 if is_varlen else k.shape[1]
    heads = k.shape[1] if is_varlen else k.shape[2]
    head_size = k.shape[2] if is_varlen else k.shape[3]

    batch_stride = k.stride()[0]
    nopacked_stride = seqlen * heads * head_size

    return batch_stride == 2 * nopacked_stride

def is_packed(q,k,is_varlen = False):
    qkvpacked = is_qkvpacked(q,is_varlen)
    kvpacked = False if qkvpacked else is_kvpacked(k,is_varlen)

    return qkvpacked, kvpacked
