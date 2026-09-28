import torch
import pytest
import argparse
import os
import yaml
import sys

from flash_attn.tuning.modules.logger import logger

from flash_attn.tuning.modules.tuning_verifier import (
    dispatch_flash_attn_verifier
)

flash_attn_fwd          = "_flash_attn_forward"
flash_attn_varlen_fwd   = "_flash_attn_varlen_forward"
flash_attn_bwd          = "_flash_attn_backward"
flash_attn_varlen_bwd   = "_flash_attn_varlen_backward"
flash_attn_fwd_kvcache  = "flash_attn_with_kvcache"

def verify_solution_bin(prob_list, solution_file, max_seqlen):

    logger.info("Running with kernel selector")
    logger.info("----------------------------")
    logger.info(f"Solution file:{solution_file}")
    os.environ["MHA_SOLUTION_PATH"] = solution_file

    for prob in prob_list:
        logger.info("----------------------------")
        logger.info(prob)
        func_name = prob["api"]
        logger.info(f"verify func name:{func_name}")
        logger.info(f"max seqlen of reference: {max_seqlen}")

        ret = dispatch_flash_attn_verifier(prob, max_seqlen)
        if not ret:
            logger.error(prob)

