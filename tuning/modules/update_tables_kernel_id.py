import flatbuffers
from flash_attn.tuning.FlashTuningData import AttentionBenchTable
from flash_attn.tuning.FlashTuningData.CppApiType import CppApiType
from flash_attn.tuning.FlashTuningData.Platform import Platform
from flash_attn.tuning.modules.flash_api_constants import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_KVCACHE
)
from flash_attn.tuning.modules.export_fb import create_attention_problem
from flash_attn.tuning.modules.export_fb import create_tuning_file_binary
from flash_attn.tuning.parse_problem_yaml import Problem
from flash_attn.tuning.modules.logger import logger
import datetime
import os


api_cvt = {
    CppApiType.FlashFwd: FLASH_FWD,
    CppApiType.FlashVarlenFwd: FLASH_VARLEN_FWD,
    CppApiType.FlashBwd: FLASH_BWD,
    CppApiType.FlashVarlenBwd: FLASH_VARLEN_BWD,
    CppApiType.FlashFwdKvcache: FLASH_KVCACHE
}

platform_cvt = {
    Platform.MXC500: "MXC500",
    Platform.MXC550: "MXC550",
}


def update_tables_kernel_id(base_file, old_kernel_id, new_kernel_id):
    with open(base_file, "rb") as f:
        base_buf = f.read()

    base_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(base_buf, 0)

    base_platform = base_table.Platform()

    # Create a new builder for the merged table
    builder = flatbuffers.Builder(1024)

    # Create a set of hash codes from new table
    hashes = set()
    problems = []

    # Process all problems from base table first
    for i in range(base_table.ProblemsLength()):
        base_problem = base_table.Problems(i)
        hash_code = base_problem.HashCode()
        hashes.add(hash_code)
        problems.append(base_problem)

    # Create problem vector
    problem_offsets = []
    count = 0
    for problem in problems:
        # Create solution
        solution = problem.Solution()
        kernel_id = solution.KernelId().decode('utf-8') if solution.KernelId() else ""
        # update kernel_id
        if kernel_id == old_kernel_id:
            kernel_id = new_kernel_id
            count += 1
        solution_dict = {
            "head_dim": solution.HeadDim(),
            "alg": solution.Alg(),
            "kernel_type": solution.KernelType(),
            "num_splits": solution.NumSplits(),
            "kernel_id": kernel_id,
            "time_us": solution.Score(),
        }

        # Create seqlens vectors
        seqlens_q = []
        seqlens_kv = []
        for i in range(problem.SeqlensQLength()):
            seqlens_q.append(problem.SeqlensQ(i))
        for i in range(problem.SeqlensKvLength()):
            seqlens_kv.append(problem.SeqlensKv(i))

        problem_params_dict = {
            "is_fp16": True if problem.Dtype() == 0 else False,
            "head_dim": problem.HeadDim(),
            "head_dim_v": problem.HeadDimV(),
            "num_heads_q": problem.NumHeadsQ(),
            "num_heads_kv": problem.NumHeadsKv(),
            "batch_size": problem.BatchSize(),
            "seqlens_q": seqlens_q,
            "seqlens_kv": seqlens_kv,
            "total_seqlens_q": problem.TotalSeqlensQ(),
            "total_seqlens_kv": problem.TotalSeqlensKv(),
            "max_seqlen_q": problem.MaxSeqlenQ(),
            "max_seqlen_kv": problem.MaxSeqlenKv(),
            "causal": problem.Causal(),
            "dropout": problem.Dropout(),
            "alibi": problem.Alibi(),
            "window_left": problem.WindowLeft(),
            "window_right": problem.WindowRight(),
            "attn_mask": problem.AttnMask(),
            "deterministic": problem.Deterministic(),
            "paged_kv": problem.PagedKv(),
            "paged_block_size": problem.PagedBlockSize(),
            "paged_num_blocks": problem.PagedNumBlocks(),
            "append_kv": problem.AppendKv(),
            "rope": problem.Rope(),
            "hash_code": problem.HashCode(),
        }

        if problem.Qkvpacked():
            problem_params_dict["qkvpacked"] = True
        if problem.Kvpacked():
            problem_params_dict["kvpacked"] = True

        problem_params_dict["api"] = api_cvt[problem.Cppapi()]

        # Convert to problem class
        prob = Problem.from_dict(problem_params_dict)

        problem_offset = create_attention_problem(builder, prob, solution_dict)
        problem_offsets.append(problem_offset)

    if count == 0:
        print("no data changed, skip")
    else:
        print(f"updated problem {count=}")
        create_tuning_file_binary(builder, problem_offsets, platform_cvt[base_platform], base_file)