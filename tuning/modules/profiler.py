import torch
from torch.profiler import profile, record_function, ProfilerActivity

def pytorch_profiler(fn, *inputs, warmups=5, repeats=500, cpu=False, verbose=True, **kwinputs):
    # Warm up
    for _ in range(warmups):
        fn(*inputs, **kwinputs)

    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(repeats):
            fn(*inputs, **kwinputs)
    if verbose:
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    return prof


def find_flash_bwd_cuda_time(prof):
    for item in prof.key_averages():
        if 'flash_bwd_dq_dk_dv' in item.key:
            return item.device_time # returns time in microseconds (μs)
    return None  # Return None if 'flash_bwd' is not found

def find_flash_fwd_cuda_time(prof):
    for item in prof.key_averages():
        if 'flash_fwd_kernel' in item.key:
            return item.device_time # returns time in microseconds (μs)
    return None  # Return None if 'flash_fwd' is not found

def find_flash_fwd_splitkv_cuda_time(prof):
    device_time = 0
    for item in prof.key_averages():
        if 'flash_fwd_splitkv_kernel' in item.key:
            device_time += item.device_time # returns time in microseconds (μs)
    return (device_time if device_time > 0 else None) # Return None if 'flash_fwd_splitkv' is not found