import pkgutil
import yaml
from flash_attn.tuning.FlashTuningData import (
    AttentionBenchTable,
    AttentionProblem,
    AttentionSolution,
)
from flash_attn.tuning.FlashTuningData.DataType import DataType
from flash_attn.tuning.FlashTuningData.CppApiType import CppApiType
from flash_attn.tuning.FlashTuningData.Platform import Platform
from flash_attn.tuning.modules.flash_api_constants import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_KVCACHE
)


def convert_api_to_fb(api):
    """
    Convert API enum from YAML to FlashTuningData CppApiType enum.
    """
    api_map = {
        "_flash_attn_forward": CppApiType.FlashFwd,
        "_flash_attn_varlen_forward": CppApiType.FlashVarlenFwd,
        "_flash_attn_backward": CppApiType.FlashBwd,
        "_flash_attn_varlen_backward": CppApiType.FlashVarlenBwd,
        "flash_attn_with_kvcache": CppApiType.FlashFwdKvcache
    }

    return api_map.get(api, CppApiType.FlashFwd)


def convert_platform_to_fb(platform):
    """
    Convert platform type string from config to FlashTuningData Platform enum.
    """
    platform_map = {
        "MXC500": Platform.MXC500,
        "MXC550": Platform.MXC550,
    }

    return platform_map.get(platform, Platform.MXC500)


def create_attention_int_vector(builder, int_list):
    AttentionProblem.AttentionProblemStartSeqlensQVector(builder, len(int_list))
    for it in reversed(int_list):
        builder.PrependInt32(it)
    return builder.EndVector()


def create_attention_solution(builder, solution):
    kernel_id_offset = builder.CreateString(solution["kernel_id"])
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, solution["head_dim"])
    AttentionSolution.AttentionSolutionAddAlg(builder, solution["alg"])
    AttentionSolution.AttentionSolutionAddKernelType(builder, solution["kernel_type"])
    AttentionSolution.AttentionSolutionAddNumSplits(builder, solution["num_splits"])
    AttentionSolution.AttentionSolutionAddScore(builder, solution['time_us'])
    AttentionSolution.AttentionSolutionAddKernelId(builder, kernel_id_offset)
    return AttentionSolution.AttentionSolutionEnd(builder)


def create_attention_problem(builder, params, best_solution):

    # Create all nested first
    seqlens_q_vector = create_attention_int_vector(builder, params["seqlens_q"])
    seqlens_kv_vector = create_attention_int_vector(builder, params["seqlens_kv"])
    solution = create_attention_solution(builder, best_solution)

    # Now start the AttentionProblem
    AttentionProblem.AttentionProblemStart(builder)
    AttentionProblem.AttentionProblemAddDtype(
        builder, DataType.float16 if params["is_fp16"] else DataType.bfloat16
    )
    AttentionProblem.AttentionProblemAddHeadDim(builder, params["head_dim"])
    AttentionProblem.AttentionProblemAddHeadDimV(builder, params["head_dim_v"])
    AttentionProblem.AttentionProblemAddNumHeadsQ(builder, params["num_heads_q"])
    AttentionProblem.AttentionProblemAddNumHeadsKv(builder, params["num_heads_kv"])
    AttentionProblem.AttentionProblemAddBatchSize(builder, params["batch_size"])
    AttentionProblem.AttentionProblemAddSeqlensQ(builder, seqlens_q_vector)
    AttentionProblem.AttentionProblemAddSeqlensKv(builder, seqlens_kv_vector)
    AttentionProblem.AttentionProblemAddTotalSeqlensQ(builder, params["total_seqlens_q"])
    AttentionProblem.AttentionProblemAddTotalSeqlensKv(builder, params["total_seqlens_kv"])
    AttentionProblem.AttentionProblemAddMaxSeqlenQ(builder, params["max_seqlen_q"])
    AttentionProblem.AttentionProblemAddMaxSeqlenKv(builder, params["max_seqlen_kv"])
    AttentionProblem.AttentionProblemAddCausal(builder, params["causal"])
    AttentionProblem.AttentionProblemAddDropout(builder, params["dropout"])
    AttentionProblem.AttentionProblemAddAlibi(builder, params["alibi"])
    if "qkvpacked" in params.__dict__.keys():
        AttentionProblem.AttentionProblemAddQkvpacked(builder, params["qkvpacked"])
    if "kvpacked" in params.__dict__.keys():
        AttentionProblem.AttentionProblemAddKvpacked(builder, params["kvpacked"])
    AttentionProblem.AttentionProblemAddWindowLeft(builder, params["window_left"])
    AttentionProblem.AttentionProblemAddWindowRight(builder, params["window_right"])
    AttentionProblem.AttentionProblemAddAttnMask(builder, params["attn_mask"])
    AttentionProblem.AttentionProblemAddDeterministic(builder, params["deterministic"])
    AttentionProblem.AttentionProblemAddPagedKv(builder, params["paged_kv"])
    AttentionProblem.AttentionProblemAddPagedBlockSize(
        builder, params["paged_block_size"]
    )
    AttentionProblem.AttentionProblemAddPagedNumBlocks(
        builder, params["paged_num_blocks"]
    )
    AttentionProblem.AttentionProblemAddAppendKv(builder, params["append_kv"])
    AttentionProblem.AttentionProblemAddRope(builder, params["rope"])
    AttentionProblem.AttentionProblemAddHashCode(builder, params["hash_code"])
    AttentionProblem.AttentionProblemAddCppapi(
        builder, convert_api_to_fb(params["api"])
    )
    AttentionProblem.AttentionProblemAddSolution(builder, solution)
    return AttentionProblem.AttentionProblemEnd(builder)


def create_tuning_file_binary(
    builder, problems, platform, output_file="flash_attn_tuning_table.bin"
):
    # Create a vector of problem offsets
    AttentionBenchTable.AttentionBenchTableStartProblemsVector(builder, len(problems))
    for problem in reversed(problems):
        builder.PrependUOffsetTRelative(problem)
    problems_vector = builder.EndVector()

    # Create a string of version
    config = yaml.safe_load(pkgutil.get_data("flash_attn.tuning", "tuner_config.yaml"))
    version = config["project"]["version"]
    version_offset = builder.CreateString(version)

    # Create an AttentionBenchTable
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems_vector)
    AttentionBenchTable.AttentionBenchTableAddPlatform(
        builder, convert_platform_to_fb(platform)
    )
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version_offset)
    tuning_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    # Finish the FlatBuffer
    builder.Finish(tuning_table)

    # Get the binary buffer
    buf = builder.Output()

    # Now you can write this buffer to a file or send it over the network
    with open(output_file, "wb") as f:
        f.write(buf)
