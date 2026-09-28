import yaml
import os
from typing import Dict, List
import inspect
from flash_attn.tuning.modules.parse_problem_list import parse_origin_to_problem

class Problem:
    def __init__(
        self,
        head_dim,
        head_dim_v,
        num_heads_q,
        num_heads_kv,
        batch_size,
        batch_size_c,
        max_seqlen_q,
        max_seqlen_kv,
        total_seqlens_q,
        total_seqlens_kv,
        seqlen_knew,
        seqlens_q,
        seqlens_kv,
        causal,
        dropout,
        alibi,
        attn_mask,
        deterministic,
        qkvpacked,
        kvpacked,
        window_left,
        window_right,
        paged_kv,
        paged_block_size,
        paged_num_blocks,
        append_kv,
        rope,
        softcap,
        is_fp16,
        api,
        hash_code,
        count,
    ):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.batch_size = batch_size
        self.batch_size_c = batch_size_c
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.total_seqlens_q = total_seqlens_q
        self.total_seqlens_kv = total_seqlens_kv
        self.seqlen_knew = seqlen_knew
        self.seqlens_q = seqlens_q
        self.seqlens_kv = seqlens_kv
        self.causal = causal
        self.dropout = dropout
        self.alibi = alibi
        self.attn_mask = attn_mask
        self.deterministic = deterministic
        self.qkvpacked = qkvpacked
        self.kvpacked = kvpacked
        self.window_left = window_left
        self.window_right = window_right
        self.paged_kv = paged_kv
        self.paged_block_size = paged_block_size
        self.paged_num_blocks = paged_num_blocks
        self.append_kv = append_kv
        self.rope = rope
        self.softcap = softcap
        self.is_fp16 = is_fp16
        self.api = api
        self.hash_code = hash_code
        self.count = count
        self.name = "problem"
        self.missing_fields = None

    @classmethod
    def from_dict(cls, data):
        """
        Parse problem instance form dict
        """
        return cls(
            head_dim=data.get("head_dim"),
            head_dim_v=data.get("head_dim_v"),
            num_heads_q=data.get("num_heads_q"),
            num_heads_kv=data.get("num_heads_kv"),
            batch_size=data.get("batch_size"),
            batch_size_c=data.get("batch_size_c"),
            max_seqlen_q=data.get("max_seqlen_q"),
            max_seqlen_kv=data.get("max_seqlen_kv"),
            total_seqlens_q=data.get("total_seqlens_q"),
            total_seqlens_kv=data.get("total_seqlens_kv"),
            seqlen_knew=data.get("seqlen_knew"),
            seqlens_q=data.get("seqlens_q"),
            seqlens_kv=data.get("seqlens_kv"),
            causal=data.get("causal"),
            dropout=data.get("dropout"),
            alibi=data.get("alibi"),
            attn_mask=data.get("attn_mask"),
            deterministic=data.get("deterministic"),
            qkvpacked=data.get("qkvpacked", False),
            kvpacked=data.get("kvpacked", False),
            window_left=data.get("window_left"),
            window_right=data.get("window_right"),
            paged_kv=data.get("paged_kv"),
            paged_block_size=data.get("paged_block_size"),
            paged_num_blocks=data.get("paged_num_blocks"),
            append_kv=data.get("append_kv"),
            rope=data.get("rope"),
            softcap=data.get("softcap"),
            is_fp16=data.get("is_fp16"),
            api=data.get("api"),
            hash_code=data.get("hash_code"),
            count = data.get("count")
        )

    @classmethod
    def check_missing_fields(cls, data):
        """
        检查传入的字典是否包含了类的所有字段，返回缺失的字段列表
        """
        # 获取类的 __init__ 方法的参数
        init_signature = inspect.signature(cls.__init__)
        # 获取所有参数的名称（排除 self）
        required_fields = list(init_signature.parameters.keys())[1:]  # 去掉第一个参数 self

        # 检查字典中是否包含所有字段
        missing_fields = [field for field in required_fields if field not in data]

        return missing_fields

    def get(self, key, default=None):
        """
        根据字段名称返回相应的值
        如果字段不存在，返回指定的默认值(None)
        """
        return getattr(self, key, default)

    def __getitem__(self, key):
        """
        实现 [] 操作符，根据字段名称返回相应的值
        如果字段不存在，抛出 KeyError
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Field '{key}' does not exist in Problem")

    def __repr__(self):
        """
        tostring
        """
        return (
            f"Problem(name={self.name},count={self.count},head_dim={self.head_dim}, head_dim_v={self.head_dim_v}, "
            f"num_heads_q={self.num_heads_q}, num_heads_kv={self.num_heads_kv}, "
            f"batch_size={self.batch_size}, batch_size_c={self.batch_size_c}, "
            f"max_seqlen_q={self.max_seqlen_q}, max_seqlen_kv={self.max_seqlen_kv}, "
            f"total_seqlens_q={self.total_seqlens_q}, total_seqlens_kv={self.total_seqlens_kv}, "
            f"seqlen_knew={self.seqlen_knew}, causal={self.causal}, dropout={self.dropout}, "
            # f"seqlens_kv={self.seqlens_kv}, seqlens_q={self.seqlens_q}, "
            f"alibi={self.alibi}, attn_mask={self.attn_mask}, deterministic={self.deterministic}, "
            f"qkvpacked={self.qkvpacked}, kvpacked={self.kvpacked}, window_left={self.window_left}, "
            f"window_right={self.window_right}, paged_kv={self.paged_kv}, "
            f"paged_block_size={self.paged_block_size}, paged_num_blocks={self.paged_num_blocks}, "
            f"append_kv={self.append_kv}, rope={self.rope}, softcap={self.softcap}, "
            f"is_fp16={self.is_fp16}, api={self.api}, hash_code={self.hash_code})"
        )

def parse_origin_problem(origin_problem_list, prob_map):
    for org_prob in origin_problem_list:
        # prob_dict = org_prob["prob"]
        origin_dict = org_prob["origin"]
        # parse the origin dict to update hashcode
        prob_dict = parse_origin_to_problem(origin_dict)
        prob_dict["count"] = org_prob["count"]
        prob_dict["name"] = org_prob["name"]

        if "hash_code" not in prob_dict:
            continue

        hash_code = prob_dict["hash_code"]
        if hash_code in prob_map:
            prob_map[hash_code]["count"] += prob_dict["count"]
        else:
            prob_map[hash_code] = prob_dict

def parse_problem(file_paths):

    prob_map = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        if isinstance(yaml_data, list):
            parse_origin_problem(yaml_data, prob_map)
        else:
            for key in yaml_data:
                if key.startswith('problem_'):
                    prob_dict = yaml_data[key]
                    prob_dict["name"] = key
                    if "count" not in prob_dict:
                        prob_dict["count"] = 1
                    if "hash_code" not in prob_dict:
                        continue
                    hash_code = prob_dict["hash_code"]
                    if hash_code in prob_map:
                        # update count
                        prob_map[hash_code]["count"] += prob_dict.get("count", 1)
                    else:
                        prob_map[hash_code] = prob_dict

    probs = []
    # parse problem from dict
    for key in prob_map:
        missing_fields = Problem.check_missing_fields(prob_map[key])
        prob = Problem.from_dict(prob_map[key])
        prob.name = f"problem_{key}"
        prob.missing_fields = missing_fields

        # update kvcache problem if max_seqlen_kv = 0
        if prob.paged_kv and prob.max_seqlen_kv == 0:
            prob.batch_size_c = prob.batch_size
            prob.max_seqlen_kv = prob.paged_block_size * prob.paged_num_blocks

        probs.append(prob)

    return probs


def parse_problem_yaml(path: str):
    problem_files = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.yaml'):
                problem_files.append(os.path.join(path, file))
    elif os.path.isfile(path):
        problem_files.append(path)
    else:
        raise ValueError(f"The path {path} is neither a directory nor a file.")

    return parse_problem(problem_files)