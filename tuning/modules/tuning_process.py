import yaml
from flash_attn.tuning.modules.dispatch_attention_api import (
    dispatch_flash_attention_api,
    set_solution_params,
)
from flash_attn.tuning.modules.kernel_registry import (
    forward_kernels_fp16,
    forward_kernels_bf16,
    forward_splitkv_kernels_fp16,
    forward_splitkv_kernels_bf16,
    backward_kernels_fp16,
    backward_kernels_bf16,
)
from flash_attn.tuning.modules.profiler import (
    find_flash_fwd_cuda_time,
    find_flash_bwd_cuda_time,
    find_flash_fwd_splitkv_cuda_time,
)
from flash_attn.tuning.modules.splits_heuristic import num_splits_heuristic
from flash_attn.tuning.FlashTuningData.KernelType import KernelType
from flash_attn.tuning.modules.logger import logger

from flash_attn.tuning.modules.flash_api_constants import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_KVCACHE,
)


# for one mha problem
def tuning_process(params, platform, config, profile_verbose = False):
    api = params.get("api")

    # Get the head dimension from params and round up to the nearest multiple of 32
    head_dim = params.get("head_dim")
    head_dim_v = params.get("head_dim_v")
    rounded_head_dim = ((head_dim + 31) // 32) * 32
    rounded_head_dim_v = ((head_dim_v + 31) // 32) * 32
    is_fp16 = params.get("is_fp16")

    if "forward" in api or "kvcache" in api:
        op_type = "fwd"
    elif "backward" in api:
        op_type = "bwd"
    else:
        raise ValueError(f"Unsupported API: {api}")

    # Read forward kernels for the rounded head dimension
    if is_fp16:
        if op_type == "fwd":
            fwd_kernels = forward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
            fwd_splitkv_kernels = []
            if head_dim == head_dim_v: # split kernel don't support head_dim != head_dim_v
                fwd_splitkv_kernels = forward_splitkv_kernels_fp16.get(
                    f"hdim{rounded_head_dim}", []
                )
            fwd_kernels = [item for item in fwd_kernels if item["hdim_v"] == rounded_head_dim_v]
        else:
            bwd_kernels = backward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
            bwd_kernels = [item for item in bwd_kernels if item["hdim_v"] == rounded_head_dim_v]
    else:
        if op_type == "fwd":
            fwd_kernels = forward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])
            fwd_splitkv_kernels = []
            if head_dim == head_dim_v: # split kernel don't support head_dim != head_dim_v
                fwd_splitkv_kernels = forward_splitkv_kernels_bf16.get(
                    f"hdim{rounded_head_dim}", []
                )
            fwd_kernels = [item for item in fwd_kernels if item["hdim_v"] == rounded_head_dim_v]
        else:
            bwd_kernels = backward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])
            bwd_kernels = [item for item in bwd_kernels if item["hdim_v"] == rounded_head_dim_v]

    if op_type == "fwd":
        logger.debug(f"Forward kernels for rounded head_dim {rounded_head_dim}:")
        for kernel in fwd_kernels:
            logger.debug(f" - {kernel['kernel_id']}")
        logger.debug(
            f"Forward-splitkv kernels for rounded head_dim {rounded_head_dim}:"
        )
        for kernel in fwd_splitkv_kernels:
            logger.debug(f" - {kernel['kernel_id']}")

    if op_type == "bwd":
        logger.debug(f"Backward kernels for rounded head_dim {rounded_head_dim}:")
        for kernel in bwd_kernels:
            logger.debug(f" - {kernel['kernel_id']}")
    logger.debug("=" * 50)

    # use flash attention's policy to determine whether to force use split kernels
    force_split_kernels = (
        "kvcache" in api or "forward" in api
    ) and params.get("paged_kv")
    logger.debug(f"Force split kernels: {force_split_kernels}")
    logger.debug("=" * 50)

    def process_fwd_kernel(kernel, alg, params):
        logger.debug(
            f"Processing fwd kernel: {kernel['kernel_id']}, {alg}"
        )
        set_solution_params(kernel["kernel_id"], KernelType().Fwd, alg)
        prof = dispatch_flash_attention_api(params, config, verbose=profile_verbose)
        if prof is None:
            logger.debug(f"kernel:{kernel['kernel_id']} running error")
            if config["debug"] != 0:
                kernel_error_info = {}
                kernel_error_info["kernel"] = kernel["kernel_id"]
                kernel_error_info["input"] = params["name"]
                with open("kernel_error.yaml","a") as f:
                    yaml.dump(kernel_error_info, f)
            return None

        device_time = find_flash_fwd_cuda_time(prof)
        if device_time is None:
            logger.warning(f"[Fwd] CUDA time is None!")
            return None

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "alg": alg,
            "num_splits": 1,
            "kernel_type": KernelType().Fwd,
            "time_us": device_time,
        }

    def process_bwd_kernel(kernel, alg, params):
        logger.debug(
            f"Processing bwd kernel: {kernel['kernel_id']}, {alg}"
        )
        set_solution_params(kernel["kernel_id"], KernelType().Bwd, alg)
        prof = dispatch_flash_attention_api(params, config, verbose=profile_verbose)
        if prof is None:
            logger.debug(f"kernel:{kernel['kernel_id']} running error")
            if config["debug"] != 0:
                kernel_error_info = {}
                kernel_error_info["kernel"] = kernel["kernel_id"]
                kernel_error_info["input"] = params["name"]
                with open("kernel_error.yaml","a") as f:
                    yaml.dump(kernel_error_info, f)
            return None

        device_time = find_flash_bwd_cuda_time(prof)
        if device_time is None:
            logger.warning(f"[Bwd] CUDA time is None!")
            return None

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "alg": alg,
            "num_splits": 1,
            "kernel_type": KernelType().Bwd,
            "time_us": device_time,
        }

    def process_fwd_splitkv_kernel(kernel, alg, params):
        api = params.get("api")
        is_decoding = max(params.get("seqlens_q")) ==  1
        batch_size = params.get("batch_size")
        num_heads = params.get("num_heads_q")
        max_seqlen_q = max(params.get("seqlens_q"))
        max_seqlen_kv = params.get("max_seqlen_kv")
        block_m = kernel["block_m"]
        block_n = kernel["block_n"]
        num_m_blocks = (max_seqlen_q + block_m - 1) // block_m
        num_n_blocks = (max_seqlen_kv + block_n - 1) // block_n
        assert (
            platform == "MXC500" or platform == "MXC550"
        ), f"Unsupported platform: {platform}. Only MXC500 or MXC550 is supported."
        num_SMs = 104
        logger.debug(f'max_seqlen_q:{max_seqlen_q},max_seqlen_kv:{max_seqlen_kv}')
        logger.debug(f'batch:{batch_size},num heads:{num_heads},num_m_block:{num_m_blocks},num_n_blocks:{num_n_blocks}')

        num_ns = config.get("num_ns",5)
        num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks, num_SMs * 2, num_n_blocks, top_n=num_ns
        )
        if ("varlen" in api) and not is_decoding:   # combine kernel don't support varlen seqlens_q
            num_splits = [1]
        logger.debug(f'prcess fwd splitkv kernel, kernel:{kernel},alg:{alg}')
        logger.debug(f'num splits:{num_splits}')
        results = []
        kernel_id = kernel["kernel_id"]
        for ns in num_splits:
            logger.debug(
                f"Processing fwd-splitkv kernel: {kernel_id}, {alg}, {ns}"
            )

            set_solution_params(
                kernel["kernel_id"], KernelType().FwdSplitkv, alg, ns
            )
            prof = dispatch_flash_attention_api(params, config, verbose=profile_verbose)
            if prof is None:
                logger.debug(f"kernel:{kernel['kernel_id']} running error")
                if config["debug"] != 0:
                    kernel_error_info = {}
                    kernel_error_info["kernel"] = kernel["kernel_id"]
                    kernel_error_info["input"] = params["name"]
                    with open("kernel_error.yaml","a") as f:
                        yaml.dump(kernel_error_info, f)
                return None

            device_time = find_flash_fwd_splitkv_cuda_time(prof)
            if device_time is None:
                logger.warning(f"[FwdSplitkv] CUDA time is None!")
            else:
                results.append(
                    {
                        "head_dim": rounded_head_dim,
                        "kernel_id": kernel_id,
                        "alg": alg,
                        "num_splits": ns,
                        "kernel_type": KernelType().FwdSplitkv,
                        "time_us": device_time,
                    }
                )
        return results

    results = []
    if op_type == "fwd":
        if not force_split_kernels:
            max_alg = 12 if not config["test_mode"] else 1
            logger.info(f"Process fwd kernel, MAX_ALG:{max_alg}")
            results = [
                result
                for result in (
                    process_fwd_kernel(kernel, alg, params)
                    for kernel in fwd_kernels
                    for alg in range(max_alg)
                )
                if result is not None
            ]

        if params.get("attn_mask") is True:
            logger.info("attn_mask is True, skip process_fwd_splitkv_kernel (not support)")
        else:
            max_alg = 6 if not config["test_mode"] else 1 # for test
            logger.info(f"Process fwd split kernel, MAX_ALG:{max_alg}")
            fwd_splitkv_results = [
                process_fwd_splitkv_kernel(kernel, alg, params)
                for kernel in fwd_splitkv_kernels
                for alg in range(max_alg)
            ]
            logger.debug(f'fwd splitkv results:{fwd_splitkv_results}')
            if fwd_splitkv_results is not None:
                for result in fwd_splitkv_results:
                    if result is not None:
                        results.extend(result)
    else:
        results = []
        for kernel in bwd_kernels:
            num_n_block = (params.max_seqlen_kv + kernel["block_n"] - 1) // kernel["block_n"]
            max_alg = 12 if num_n_block % 2 == 0 else 6
            if params["deterministic"]:
                max_alg = 1
            if config["test_mode"]: # for test
                max_alg = 1
            logger.info(f"Process bwd kernel, MAX_ALG:{max_alg}, derterministic:{params['deterministic']}")
            for alg in range(max_alg):
                result = process_bwd_kernel(kernel,alg,params)
                if result is not None:
                    results.append(result)

    return results
