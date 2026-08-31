from copy import copy
from enum import Enum, auto
from itertools import count

from lazy import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

"""
Sequence represents a single request to the LLM. It contains the token ids of the prompt and the generated tokens, 
as well as the status of the sequence (waiting, running, or finished). It also contains information about the number of tokens, 
the number of prompt tokens, the number of cached tokens, and the number of scheduled tokens. 
The block table is used to store the cached blocks of tokens for efficient memory management.
"""
class Sequence:
    block_size = 256
    counter = count() # globla seq id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]
