from collections import deque

from lazy.config import Config
from lazy.engine.sequence import Sequence, SequenceStatus
from lazy.utils.logger import init_logger

import nvtx

logger = init_logger(__name__)


class Scheduler:

    def __init__(self, config: Config, eos_token_id: int | None = None):
        self.eos_token_id = eos_token_id
        self.waiting: deque[Sequence] = deque() # FIFO
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self):
        scheduled_seqs = []

        # prefill 一次只取一条：多条打平成一个 batch 时 causal mask 会互相看见
        if self.waiting:
            with nvtx.annotate('prefill_scheduler', color="red"):
                seq = self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append(seq)
                return scheduled_seqs, True  # True for is prefill

        # decode
        with nvtx.annotate('decode_scheduler', color="red"):
            scheduled_seqs = list(self.running)
            for seq in scheduled_seqs:
                seq.is_prefill = False
            return scheduled_seqs, False # False for is decode

    def postprocess(self, seqs, output_token_ids, is_prefill):
        finished = []
        for seq, token_id in zip(seqs, output_token_ids):
            seq.append_token(token_id)
            hit_max = seq.num_tokens - seq.num_prompt_tokens >= seq.max_tokens
            hit_eos = (not seq.ignore_eos) and self.eos_token_id is not None and token_id == self.eos_token_id
            if hit_max or hit_eos:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
                self.finished.append(seq)
                reason = "hit_max" if hit_max else "hit_eos"
                seq.finish_reason = reason
                finished.append(seq)
                logger.info("seq %d finished (%s), generated %d tokens",
                            seq.seq_id, reason, seq.num_tokens - seq.num_prompt_tokens)
        return finished
