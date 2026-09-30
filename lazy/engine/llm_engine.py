"""LLM 引擎：后台线程驱动 step 循环，请求动态加入。

引擎线程独占 scheduler / model_runner / kv_caches；提交线程只通过 submit
把请求放进 _pending，再用 RequestHandle 阻塞等结果。
"""

from collections import deque
import threading
import time

import nvtx
from transformers import AutoTokenizer

from lazy.config import Config
from lazy.engine.model_runner import ModelRunner
from lazy.engine.scheduler import Scheduler
from lazy.engine.sequence import Sequence
from lazy.utils.logger import init_logger

logger = init_logger(__name__)


class RequestHandle:
    """单次请求的结果句柄。

    引擎线程完成时写结果，提交线程用 result() 阻塞等待。
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.text: str | None = None
        self.error: str | None = None
        self.submit_time = time.perf_counter()

    def set_result(self, text: str) -> None:
        """写入正常结果并唤醒等待者。"""
        self.text = text
        self._event.set()

    def set_error(self, error: str) -> None:
        """写入错误并唤醒等待者。"""
        self.error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> str:
        """阻塞等待结果。

        Args:
            timeout: 最长等待秒数，None 表示一直等。

        Returns:
            生成的文本。

        Raises:
            TimeoutError: 超时还没结果。
            RuntimeError: 引擎侧报了错。
        """
        if not self._event.wait(timeout):
            raise TimeoutError("request timed out")
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.text


class LLMEngine:
    """推理引擎的对外入口。"""

    def __init__(self, model_path: str):
        """加载模型、建调度器，并起后台引擎线程。

        Args:
            model_path: 本地模型目录。
        """
        logger.info("initializing engine, model: %s", model_path)
        config = Config(model_path)

        logger.info("loading model weights (this may take a while)")
        self.model_runner = ModelRunner(config)

        eos_token_id = config.hf_config.eos_token_id
        # Qwen3-Base 的 eos 可能是 list，取第一个。
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        self.scheduler = Scheduler(config, eos_token_id=eos_token_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model,
                                                       use_fast=True)
        logger.info("engine ready")

        # 提交口：提交线程只碰 _pending，引擎线程独占其余内部状态。
        self._cond = threading.Condition()
        self._pending: deque[Sequence] = deque()
        self._handles: dict[int, RequestHandle] = {}
        self._engine_thread = threading.Thread(target=self._engine_loop,
                                               daemon=True,
                                               name="engine-loop")
        self._engine_thread.start()

    def submit(self, prompt: str | list[int], sampling_params) -> RequestHandle:
        """提交一个请求，立即返回句柄。

        引擎线程会在后续 step 里把请求接进调度队列。

        Args:
            prompt: 字符串或已经 tokenize 好的 token id 列表。
            sampling_params: 采样参数。

        Returns:
            用于取结果的 RequestHandle。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        handle = RequestHandle()
        with self._cond:
            self._handles[seq.seq_id] = handle
            self._pending.append(seq)
            self._cond.notify()
        logger.info("request added: seq %d, %d prompt tokens", seq.seq_id,
                    seq.num_prompt_tokens)
        return handle

    def _engine_loop(self) -> None:
        """引擎主循环：排空新请求，跑一步，空闲就等条件变量。"""
        while True:
            with self._cond:
                while not self._pending and self.scheduler.is_finished():
                    self._cond.wait()
                while self._pending:
                    self.scheduler.add(self._pending.popleft())
            if self.scheduler.is_finished():
                continue
            try:
                self.step()
            except Exception as err:
                logger.exception(
                    "engine step failed, failing all in-flight requests")
                self._fail_all(f"engine step failed: {err}")

    def _fail_all(self, error: str) -> None:
        """step 出错时把在飞请求全部置错并清空队列"""
        with self._cond:
            while self._pending:
                seq = self._pending.popleft()
                self._handles.pop(seq.seq_id, None)
        for seq in list(self.scheduler.waiting) + list(self.scheduler.running):
            handle = self._handles.pop(seq.seq_id, None)
            if handle is not None:
                handle.set_error(error)
            self.model_runner.free_cache(seq.seq_id)
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()

    @nvtx.annotate("step", color="red")
    def step(self) -> None:
        """调度、跑模型、回收结果，完成一步。"""
        seqs, is_prefill = self.scheduler.schedule()

        out_token_id = self.model_runner.run(seqs, is_prefill)

        logger.debug("step: %d seqs, is_prefill=%s, out token: %s", len(seqs),
                     is_prefill, self.tokenizer.decode(out_token_id))

        finished = self.scheduler.postprocess(seqs, out_token_id, is_prefill)
        for seq in finished:
            self.model_runner.free_cache(seq.seq_id)
            text = self.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens:])
            handle = self._handles.pop(seq.seq_id, None)
            if handle is not None:
                handle.set_result(text)
                num_new_tokens = seq.num_tokens - seq.num_prompt_tokens
                elapsed = time.perf_counter() - handle.submit_time
                logger.info(
                    "generate done: seq %d, %d tokens in %.2fs (%.2f tok/s)",
                    seq.seq_id,
                    num_new_tokens,
                    elapsed,
                    num_new_tokens / max(elapsed, 1e-9),
                )
            self.scheduler.finished.remove(seq)

    def is_finished(self) -> bool:
        """所有请求都跑完了吗。"""
        return self.scheduler.is_finished()

    def generate(self, prompts: str | list[str], sampling_params) -> list[str]:
        """同步批量接口：全部提交后按序等结果。

        Args:
            prompts: 一个 prompt 或一批 prompt。
            sampling_params: 采样参数。

        Returns:
            与 prompts 一一对应的生成文本列表。
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        handles = [self.submit(prompt, sampling_params) for prompt in prompts]
        return [handle.result() for handle in handles]
