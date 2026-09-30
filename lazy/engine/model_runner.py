"""模型执行器：准备输入、跑 forward、采样下一个 token。"""

import torch

from lazy.cache import BatchedDecodeCache
from lazy.cache import KVCache
from lazy.config import Config
from lazy.engine.sequence import Sequence
from lazy.layers.sampler import Sampler
from lazy.models.qwen3 import Qwen3ForCausalLM
from lazy.utils.loader_weight import load_weights
from lazy.utils.logger import init_logger

logger = init_logger(__name__)


class ModelRunner:
    """包住模型权重和 KV cache，负责单步的输入准备、forward 和采样。

    Attributes:
        kv_caches: seq_id 到该请求独立 KV cache 的映射。
    """

    def __init__(self, config: Config):
        """加载模型权重并初始化采样器。

        Args:
            config: 引擎配置。
        """
        logger.info("building model on cuda (bf16)")
        model = Qwen3ForCausalLM(config.hf_config)
        self.model = model.to(dtype=torch.bfloat16).cuda()
        logger.info("loading weights from %s", config.model)
        loaded = load_weights(self.model, config.model)
        assert loaded, "weight missing, please check the model path or hf name"
        self.sampler = Sampler()
        self.num_layers = config.hf_config.num_hidden_layers
        # seq_id -> 每个请求独立的 KV cache
        self.kv_caches: dict[int, KVCache] = {}
        logger.info("model runner ready")

    def prepare_prefill(
            self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """把 prefill 输入拼成 [1, total_len] 的 token / position 张量。"""
        input_ids = []
        positions = []
        for seq in seqs:
            input_ids.extend(seq.token_ids)
            positions.extend(range(0, len(seq)))

        input_ids = torch.tensor([input_ids],
                                 dtype=torch.int64,
                                 pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([positions],
                                 dtype=torch.int64,
                                 pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions

    def prepare_decode(
            self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """把 decode 输入摆成 [B, 1] 的 token / position 张量。"""
        input_ids = []
        positions = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64,
            pin_memory=True).cuda(non_blocking=True).unsqueeze(1)
        positions = torch.tensor(
            positions, dtype=torch.int64,
            pin_memory=True).cuda(non_blocking=True).unsqueeze(1)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        """收集各 seq 的采样温度，返回 [B] 的张量。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures,
                                    dtype=torch.float32,
                                    pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """跑一次 forward。"""
        return self.model(
            input_ids,
            positions,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )

    def free_cache(self, seq_id: int) -> None:
        """请求结束后释放它那一份 KV cache。"""
        self.kv_caches.pop(seq_id, None)

    def _build_pad_mask(self, lengths: list[int], max_len: int) -> torch.Tensor:
        """构造 decode 拼 batch 的 padding mask。

        拼接视图 kv_len = max_len + 1（最后一列是本步新 token），
        返回 [B, 1, 1, max_len + 1]，True 表示挡住。

        Args:
            lengths: 各 seq 的历史长度（不含本步新 token）。
            max_len: 历史长度的最大值。

        Returns:
            [B, 1, 1, max_len + 1] 的 bool mask。
        """
        col = torch.arange(max_len + 1, device="cuda")
        lengths_t = torch.tensor(lengths, dtype=torch.int64, device="cuda")
        # 历史列按各自长度可见，新 token 列对所有 seq 都可见。
        valid = (col[None, :] < lengths_t[:, None]) | (col[None, :] == max_len)
        return ~valid[:, None, None, :]

    def _sample(self, logits: torch.Tensor,
                temperatures: torch.Tensor) -> list[int]:
        """按温度采样下一个 token，全 0 温度走纯 greedy。

        Args:
            logits: 形状 [B, vocab] 的 logits。
            temperatures: 形状 [B] 的温度。

        Returns:
            形状 [B] 的 token id 列表。
        """
        greedy = temperatures == 0
        if bool(greedy.all()):
            return logits.argmax(dim=-1).tolist()
        if bool(greedy.any()):
            # 混合温度：greedy 行先按占位温度走 Sampler，再覆盖成 argmax。
            placeholder = torch.ones_like(temperatures)
            tokens = self.sampler(
                logits, torch.where(greedy, placeholder, temperatures))
            tokens[greedy] = logits[greedy].argmax(dim=-1)
            return tokens.tolist()
        return self.sampler(logits, temperatures).tolist()

    def _run_seq(self, seq: Sequence, is_prefill: bool) -> int:
        """跑单条序列的一步，返回采出的 token id。"""
        if is_prefill:
            self.kv_caches[seq.seq_id] = KVCache(self.num_layers, device="cuda")
            input_ids, positions = self.prepare_prefill([seq])
            logger.debug("prefill: seq %d, %d tokens", seq.seq_id,
                         input_ids.shape[1])
        else:
            input_ids, positions = self.prepare_decode([seq])
        temperatures = self.prepare_sample([seq])
        logits = self.run_model(input_ids, positions,
                                self.kv_caches[seq.seq_id])
        return self._sample(logits[:, -1, :], temperatures)[0]

    def _run_decode_batch(self, seqs: list[Sequence]) -> list[int]:
        """把多条 decode 合成一个 batch 跑，返回各自的 token id。"""
        logger.debug("decode: %d seqs batched", len(seqs))
        input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        caches = [self.kv_caches[seq.seq_id] for seq in seqs]
        batched_cache = BatchedDecodeCache(caches)
        pad_mask = self._build_pad_mask(batched_cache.lengths,
                                        batched_cache.max_len)
        logits = self.run_model(input_ids,
                                positions,
                                batched_cache,
                                attention_mask=pad_mask)
        return self._sample(logits[:, -1, :], temperatures)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """跑一步，prefill 逐条、decode 拼 batch。"""
        if is_prefill:
            return [self._run_seq(seq, True) for seq in seqs]
        return self._run_decode_batch(seqs)
