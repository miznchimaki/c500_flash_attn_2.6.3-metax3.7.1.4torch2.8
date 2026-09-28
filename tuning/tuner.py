import yaml
import os
import datetime
import flatbuffers
from flash_attn.tuning.modules.tuning_process import tuning_process
from flash_attn.tuning.modules.export_fb import (
    create_attention_problem,
    create_tuning_file_binary,
)
from flash_attn.tuning.modules.export_yaml import append_results_to_yaml
from flash_attn.tuning.modules.kernel_registry import register_kernels
from flash_attn.tuning.modules.parse_problem_list import parse_probelm
from flash_attn.tuning.parse_problem_yaml import parse_problem_yaml
from tqdm import tqdm
import sys
from flash_attn.tuning.modules.logger import logger,LOG_LEVELS
import pkgutil


def kernel_candidate():
    return yaml.safe_load(pkgutil.get_data(__package__, "kernel_traits_candidates.yaml"))

def tuning_version():
    config = yaml.safe_load(pkgutil.get_data(__package__, "tuner_config.yaml"))
    return config["project"]["version"]

def find_best_solution(solutions):
    if not solutions:
        return {}  # Return an empty dict if results is empty

    return min(solutions, key=lambda x: x.get("time_us", float("inf")))

def process_problem(problem, output_file, builder, platform, config, profile_verbose = False):

    results = tuning_process(problem, platform, config, profile_verbose)
    if not results:
        logger.warning(f"No valid solution found for case {problem.name}. Skipping.")
        return None
    best_solution = find_best_solution(results)

    append_results_to_yaml(
        problem,
        results,
        best_solution,
        output_file,
    )
    problem = create_attention_problem(
        builder, problem, best_solution
    )
    return problem

def flash_tuning(input_path, output_dir, solution_name, config, total = None):

    if config["debug"] != 0:
        logger.setLevel(LOG_LEVELS.get("DEBUG"))

    os.makedirs(output_dir, exist_ok=True)

    # parse prob list from yaml file or dir of problem files
    prob_list = parse_problem_yaml(input_path)

    logger.info(f"out dir:{output_dir}")

    # Load config
    tuner_config = yaml.safe_load(pkgutil.get_data(__package__, "tuner_config.yaml"))["config"]
    max_seqlen = tuner_config.get("max_seqlen")
    test_mode = tuner_config.get("test_mode",False)

    # Add tuner_config to config(user)
    config["max_seqlen"] = max_seqlen
    config["test_mode"] = test_mode

    # Get kernel_tratis_candidates.yaml data and parse to dict
    traits_candidates = yaml.safe_load(pkgutil.get_data(__package__, "kernel_traits_candidates.yaml"))
    if traits_candidates is not None:
        logger.info("Registering kernels from kernel traits candidates.")
        register_kernels(traits_candidates)
    else:
        logger.error("Error: kernel_traits file not found or not specified in config.")
        sys.exit(1)

    logger.info("=" * 50 + "\n")

    # set env variables for logging
    os.environ["KINETO_LOG_LEVEL"] = "5"  # disable kineto logging
    # os.environ["MHA_LOG_ENABLE"] = "1"
    # os.environ["MHA_LOG_LEVEL"] = "3"
    # os.environ["MHA_LOG_OUTPUT"] = "file"

    platform = "MXC500"

    builder = flatbuffers.Builder(1024)
    all_attention_problems = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    profile_verbose = bool(config["verbose"])

    output_file = os.path.join(
        output_dir,
        f"{solution_name}_results_{timestamp}.yaml",
    )

    if total is None:
        prob_len = len(prob_list)
    else:
        prob_len = total

    logger.info(f"Starting benchmark")

    with tqdm(
        enumerate(prob_list),
        desc="Processing problem",
        total=prob_len,
    ) as pbar:
        for idx, problem in pbar:
            logger.info("Current problem:")
            logger.info(problem)
            logger.info("=" * 50)
            pbar.set_postfix({"Case": idx})
            result = process_problem(problem, output_file, builder, platform, config, profile_verbose = profile_verbose)
            if result is None:
                continue
            all_attention_problems.append(result)
            if idx >= prob_len:
                logger.warning(f"Test model and only process some of the problems.processed/total:{prob_len}/{len(prob_list)}")
                break

    tuning_file = f"{output_dir}/flash_attn_tuning_{platform}_{solution_name}_{timestamp}.bin"

    create_tuning_file_binary(builder, all_attention_problems, platform, tuning_file)

    logger.info(f"Benchmark table written to: {tuning_file}")

    return prob_list, tuning_file, max_seqlen

