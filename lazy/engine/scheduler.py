from collections import deque

from lazy.config import Config
from lazy.engine.sequence import Sequence, SequenceStatus

class Scheduler:

    def __init__(self, config: Config):
        
        self.waiting: deque[Sequence] = deque() # FIFO
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self):
        scheduled_seqs = []

        # prefill
        while self.waiting:
            seq = self.waiting[0]
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # True for is prefill

        # decode
        while self.running:
            seq = self.running.popleft()
            seq.is_prefill = False
            scheduled_seqs.append(seq)
        return scheduled_seqs, False # False for is decode
            