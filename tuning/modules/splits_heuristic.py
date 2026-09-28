import math
from typing import List

def num_splits_heuristic(batch_nheads_mblocks: int, num_SMs: int, num_n_blocks: int, max_splits: int = 128, efficiency_threshold: float = 0.85, top_n: int = 5) -> List[int]:
    """
    Find the top n number of splits that maximize the occupancy.
    
    For example, if we have batch * n_heads = 48 and we have 108 SMs, having 2 splits
    (efficiency = 0.89) is better than having 3 splits (efficiency = 0.67). However,
    we also don't want too many splits as that would incur more HBM reads/writes.
    
    So we find the best efficiency, then find the N smallest number of splits that gets
    the specified percentage (default 85%) of the best efficiency.
    """
    # If we have enough to almost fill the SMs, then just use 1 split
    if batch_nheads_mblocks >= 0.8 * num_SMs:
        return [1]
    
    max_splits = min(max_splits, num_SMs, num_n_blocks)
    max_efficiency = 0.0
    efficiency: List[float] = []
    
    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b
    
    def is_split_eligible(num_splits: int) -> bool:
        """
        Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
        we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
        (i.e. it's 11 splits anyway).
        So we check if the number of blocks per split is the same as the previous num_splits.
        """
        return num_splits == 1 or ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)
    
    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits):
            efficiency.append(0.0)
        else:
            n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs
            eff = n_waves / math.ceil(n_waves)
            if eff > max_efficiency:
                max_efficiency = eff
            efficiency.append(eff)
    
    eligible_splits = []
    for num_splits in range(1, max_splits + 1):
        if is_split_eligible(num_splits) and efficiency[num_splits - 1] >= efficiency_threshold * max_efficiency:
            eligible_splits.append(num_splits)
    
    # Return the first top_n eligible splits (by sequence)
    return eligible_splits[:top_n]