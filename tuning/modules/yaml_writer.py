import os
import yaml
import atexit
from datetime import datetime
from typing import Dict, List, Optional

from flash_attn.tuning.modules.problem_collection import problem_collection

class YamlWriter:
    _instance: Optional['YamlWriter'] = None
    _buffer: List[Dict] = []
    _hash_code_buffer: Dict = {}
    _prob_index: int = 0
    _buffer_count: int = 10  # Maximum capacity of the buffer
    _file_path: Optional[str] = None
    _prob_map: Dict = {}

    def __new__(cls, *args, **kwargs):
        # If the environment variable does not exist, do not create an instance
        if 'MHA_INPUT_COLLECTION' not in os.environ:
            return None

        # Singleton pattern: Ensure only one instance exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Get the directory from the environment variable
        output_dir = os.environ['MHA_INPUT_COLLECTION']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)

        # Construct the file path
        pid = os.getpid()
        self._file_path = os.path.join(output_dir, f'problem_{pid}.yaml')

        header = {
            'version': '1.0',
            'type': 'pyapi',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self._file_path, 'a') as f:
            yaml.dump(header, f, default_flow_style=False)

        # Register the exit handler function
        atexit.register(self._flush_buffer_on_exit)

    def write(self, func_name, is_bf16, data: Dict):
        """Add data to the buffer. If the buffer is full, write to the file."""
        prob_dict = problem_collection(func_name,is_bf16,data)

        hash_code = prob_dict["hash_code"]
        if hash_code in self._prob_map:
            self._prob_map[hash_code]["count"] += 1
        else:
            prob_dict["count"] = 1
            self._prob_map[hash_code] = prob_dict

            # self._buffer.append(prob_dict)
            if len(self._prob_map) >= self._buffer_count:
                self._flush_buffer()

    def is_packed(self, q,k,is_varlen = False):
        qkvpacked = self._is_qkvpacked(q,is_varlen)
        kvpacked = False if qkvpacked else self._is_kvpacked(k,is_varlen)

        return qkvpacked, kvpacked

    def _is_qkvpacked(self, q, is_varlen = False):
        seqlen = 1 if is_varlen else q.shape[1]
        heads = q.shape[1] if is_varlen else q.shape[2]
        head_size = q.shape[2] if is_varlen else q.shape[3]

        batch_stride = q.stride()[0]
        nopacked_stride = seqlen * heads * head_size

        return batch_stride == 3 * nopacked_stride

    def _is_kvpacked(self, k, is_varlen = False):
        seqlen = 1 if is_varlen else k.shape[1]
        heads = k.shape[1] if is_varlen else k.shape[2]
        head_size = k.shape[2] if is_varlen else k.shape[3]

        batch_stride = k.stride()[0]
        nopacked_stride = seqlen * heads * head_size

        return batch_stride == 2 * nopacked_stride

    def _flush_buffer(self):
        """Write the data in the buffer to the YAML file."""
        if not self._file_path :
            return

        probs = []

        for key in self._prob_map:
            probs.append({f'problem_{self._prob_index}': self._prob_map[key]})
            self._prob_index += 1

        # Write to the file
        with open(self._file_path, 'a') as f:
            for prob in probs:
                yaml.dump(prob, f, default_flow_style=False)
        print(f"Flushed buffer to file: {self._file_path}")

        # Clear the buffer
        self._prob_map.clear()

    def _flush_buffer_on_exit(self):
        """Write the buffer data to the file when the program exits."""
        print("Program is exiting, flushing buffer data to file...")
        self._flush_buffer()

    def __del__(self):
        """Destructor to ensure buffer data is written when the object is destroyed."""
        self._flush_buffer()