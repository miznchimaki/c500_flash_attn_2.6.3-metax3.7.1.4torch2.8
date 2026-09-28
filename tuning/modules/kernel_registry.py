# Dictionaries to store forward and backward kernel information
# Key: head dimension
# Value: list of dicts
forward_kernels_fp16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[], "hdim512":[]}
forward_kernels_bf16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[], "hdim512":[]}
forward_splitkv_kernels_fp16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[], "hdim512":[]}
forward_splitkv_kernels_bf16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[], "hdim512":[]}
backward_kernels_fp16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[]}
backward_kernels_bf16 = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[]}

import yaml
from flash_attn.tuning.modules.logger import logger

def register_kernels(traits):
    # with open(yaml_file, 'r') as file:
    #     traits = yaml.safe_load(file)

    # Get the list of forward kernels
    fwd_kernels = traits.get('fwd', [])
    if not fwd_kernels:
        logger.warning("No forward kernels found in YAML file")
    # Get the list of forward-splitkv kernels
    fwd_splitkv_kernels = traits.get('fwd_split', [])
    if not fwd_splitkv_kernels:
        logger.warning("No forward-splitkv kernels found in YAML file")
    # Get the list of backward kernels
    bwd_kernels = traits.get('bwd', [])
    if not bwd_kernels:
        logger.warning("No backward kernels found in YAML file")

    # Register the kernels
    for kernel in fwd_kernels:
        hdim = f"hdim{kernel['hdim_qk']}"
        # Verify that the hdim is valid
        if hdim not in forward_kernels_fp16:
            raise ValueError(f"Invalid head dimension: {hdim}")

        kernel_traits = {
                'kernel_id': kernel['kernel_id'],
                'block_m': kernel['block_m'],
                'block_n': kernel['block_n'],
                'num_warps': kernel['k_nwarps'],
                "hdim_v": kernel['hdim_v'],
                'is_q_in_regs': kernel['Is_Q_in_regs'],
                'share_q_k_smem': kernel['Share_Q_K_smem']
            }

        if kernel['dtype'] == 'float16':
            forward_kernels_fp16[hdim].append(kernel_traits)
        elif kernel['dtype'] == 'bfloat16':
            forward_kernels_bf16[hdim].append(kernel_traits)

    for kernel in fwd_splitkv_kernels:
        hdim = f"hdim{kernel['hdim_qk']}"
        # Verify that the hdim is valid
        if hdim not in forward_splitkv_kernels_fp16:
            raise ValueError(f"Invalid head dimension: {hdim}")

        kernel_traits = {
            'kernel_id': kernel['kernel_id'],
            'block_m': kernel['block_m'],
            'block_n': kernel['block_n'],
            'num_warps': kernel['k_nwarps'],
            "hdim_v": kernel['hdim_v'],
            'is_q_in_regs': kernel['Is_Q_in_regs'],
            'share_q_k_smem': kernel['Share_Q_K_smem']
        }

        if kernel['dtype'] == 'float16':
            forward_splitkv_kernels_fp16[hdim].append(kernel_traits)
        elif kernel['dtype'] == 'bfloat16':
            forward_splitkv_kernels_bf16[hdim].append(kernel_traits)

    for kernel in bwd_kernels:
        hdim = f"hdim{kernel['hdim_qk']}"
        # Verify that the hdim is valid
        if hdim not in backward_kernels_fp16:
            raise ValueError(f"Invalid head dimension: {hdim}")

        kernel_traits = {
            'kernel_id': kernel['kernel_id'],
            'block_m': kernel['block_m'],
            'block_n': kernel['block_n'],
            'num_warps': kernel['k_nwarps'],
            "hdim_v": kernel['hdim_v'],
            'atom_layout_mdq': kernel['atom_layout_mdq'],
            'atom_layout_msdp': kernel['atom_layout_msdp'],
            'atom_layout_ndkv': kernel['atom_layout_ndkv'],
            'is_k_in_regs': kernel['is_k_in_regs'],
            'is_v_in_regs': kernel['is_v_in_regs'],
            'no_double_buffer': kernel['no_double_buffer']
        }

        if kernel['dtype'] == 'float16':
            backward_kernels_fp16[hdim].append(kernel_traits)
        elif kernel['dtype'] == 'bfloat16':
            backward_kernels_bf16[hdim].append(kernel_traits)
