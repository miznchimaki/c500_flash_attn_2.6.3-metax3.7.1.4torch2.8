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


def merge_tables(base_file, new_file, merged_file, merge_mode="better"):
    with open(base_file, "rb") as f:
        base_buf = f.read()

    with open(new_file, "rb") as f:
        new_buf = f.read()

    base_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(base_buf, 0)
    new_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(new_buf, 0)

    # Check if versions match
    base_version = base_table.Version().decode('utf-8') if base_table.Version() else None
    new_version = new_table.Version().decode('utf-8') if new_table.Version() else None

    if base_version != new_version:
        raise ValueError(f"Version mismatch: base={base_version}, new={new_version}")

    # Check if platforms match
    base_platform = base_table.Platform()
    new_platform = new_table.Platform()

    if base_platform != new_platform:
        raise ValueError(
            f"Platform mismatch: base={PLATFORM_MAP[base_platform]}, new={PLATFORM_MAP[new_platform]}"
        )

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

    # Process all problems from new table
    for i in range(new_table.ProblemsLength()):
        new_problem = new_table.Problems(i)
        hash_code = new_problem.HashCode()

        if hash_code in hashes:
            # Find the corresponding problem in the base table
            for index, base_problem in enumerate(problems):
                if base_problem.HashCode() == hash_code:
                    logger.info(f"{hash_code}: base solution time {base_problem.Solution().Score()}, new solution time {new_problem.Solution().Score()}")
                    # if merge_mode == "new", update directly
                    # else, Compare scores(time) and update if the new solution is better
                    if merge_mode == "new" or new_problem.Solution().Score() < base_problem.Solution().Score():
                        problems.pop(index)
                        problems.append(new_problem)
                        logger.info("Update the base solution to new solution!")
                    else:
                        logger.info("Skip update!")
                    break
        else:
            # Add new problem if the hash code does not exist in the base table
            hashes.add(hash_code)
            problems.append(new_problem)
            logger.info(f"{hash_code}: New solution added!")

    # Create problem vector
    problem_offsets = []
    for problem in problems:
        # Create solution
        solution = problem.Solution()
        solution_dict = {
            "head_dim": solution.HeadDim(),
            "alg": solution.Alg(),
            "kernel_type": solution.KernelType(),
            "num_splits": solution.NumSplits(),
            "kernel_id": solution.KernelId().decode('utf-8') if solution.KernelId() else "",
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

    create_tuning_file_binary(builder, problem_offsets, platform_cvt[new_platform], merged_file)


def auto_merge(tables_dir):
    platform = "MXC500"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    merge_file = f"{tables_dir}/flash_attn_merge_{platform}_{timestamp}.bin"

    # Find all the files starting with "flash_attn_tuning" in the output directory
    files = [
        os.path.join(tables_dir, f)
        for f in os.listdir(tables_dir)
        if f.startswith("flash_attn_tuning")
    ]

    # Sort the files by the timestamp at the end of the filename
    files.sort(key=os.path.getctime)

    logger.info("Files for merging:")
    for file in files:
        logger.info("  * " + file)

    if len(files) < 2:
        logger.warning(f"Skipping merging because at least two files are required.")
        return

    base_file = files[0]

    for i in range(1, len(files)):
        output_file = merge_file if i == len(files) - 1 else f"{tables_dir}/temp_merged_{i}.bin"
        merge_tables(base_file, files[i], output_file, "better")
        base_file = output_file

    logger.info(f"Merged table written to: {merge_file}")
