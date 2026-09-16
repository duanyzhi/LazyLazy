import os
from collections import OrderedDict

import torch

from lazy.cache import BlockPrefixCache, hash_block


class KVCacheManager(BlockPrefixCache):
    """Tiered (SSD -> CPU -> HBM) block-granular prefix cache.

    Extends BlockPrefixCache with a three-tier storage for each cached block:
      - "hbm": KV tensors on the accelerator.
      - "cpu": KV tensors in host RAM.
      - "ssd": KV tensors serialized to disk (one torch file per block).

    HBM residency is bounded by `hbm_budget_bytes`. When it is None the budget is
    derived from the currently free HBM (`torch.cuda.mem_get_info`) minus
    `hbm_reserve_bytes`, i.e. the cache is sized by the remaining buffer. Blocks are
    evicted HBM -> CPU and (when `cpu_budget_bytes` is set) CPU -> SSD. On a hit, a
    block is pulled back up the tiers (SSD -> CPU -> HBM) before use, so a repeated
    request can be served straight from SSD.

    Eviction order is controlled by `eviction_policy`:
      - "lru": pure LRU (the component-local policy Mosaic identifies as the root of
        the attribution mismatch for agent programs).
      - "liveness": Mosaic-style liveness-anchored eviction. Each block tracks the
        programs that used it via `store(..., program_id=...)`, `suspend`, and
        `terminate`. Blocks with a running program are never evicted; the rest are
        evicted by increasing live-program reference count, with LRU as the
        intra-tier tie-breaker. Without any program state every block has zero
        references, so this degenerates to plain LRU.
    """

    def __init__(self, num_layers, block_size=16, device="cuda",
                 hbm_budget_bytes=None, cpu_budget_bytes=None, ssd_dir=None,
                 hbm_reserve_bytes=1 << 30, eviction_policy="liveness"):
        super().__init__(num_layers, block_size=block_size, device=device)
        self.ssd_dir = ssd_dir or "/tmp/lazy_kv_cache"
        os.makedirs(self.ssd_dir, exist_ok=True)
        self.lru = OrderedDict()  # block hash -> None, oldest first
        self.hbm_bytes = 0
        self.cpu_bytes = 0
        self.cpu_budget_bytes = cpu_budget_bytes  # None = unlimited
        self.eviction_policy = eviction_policy  # "lru" | "liveness"
        # Mosaic AMS program state: block hash -> {program_id: "running"|"suspended"}
        self.block_progs = {}
        self.prog_blocks = {}  # program_id -> set(block hash)
        if hbm_budget_bytes is not None:
            self.hbm_budget_bytes = hbm_budget_bytes
        else:
            free, _ = torch.cuda.mem_get_info()
            self.hbm_budget_bytes = max(0, free - hbm_reserve_bytes)

    @staticmethod
    def _kv_bytes(kv):
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)

    # -- Mosaic AMS program state -------------------------------------------------
    def begin_program(self, program_id):
        """Register a live agent program (no-op if already tracked)."""
        if program_id is None:
            return
        self.prog_blocks.setdefault(program_id, set())

    def _mark_running(self, h, program_id):
        self.begin_program(program_id)
        self.block_progs.setdefault(h, {})[program_id] = "running"
        self.prog_blocks[program_id].add(h)

    def suspend(self, program_id):
        """Running -> Suspended: the program's LLM invocation finished but the
        program is still live and may reuse its blocks on a later turn."""
        for h in self.prog_blocks.get(program_id, ()):
            progs = self.block_progs.get(h)
            if progs is not None and progs.get(program_id) == "running":
                progs[program_id] = "suspended"

    def terminate(self, program_id):
        """Running/Suspended -> Dead: drop all state entries for the program."""
        for h in self.prog_blocks.pop(program_id, ()):
            progs = self.block_progs.get(h)
            if progs is not None:
                progs.pop(program_id, None)
                if not progs:
                    self.block_progs.pop(h, None)

    def ref_count(self, h):
        """Number of live (running or suspended) programs referencing a block."""
        return len(self.block_progs.get(h, {}))

    def _ensure_hbm(self, h):
        entry = self.blocks[h]
        if entry["tier"] == "hbm":
            self.lru.move_to_end(h)
            return
        if entry["tier"] == "cpu":
            kv = [(k.to("cuda"), v.to("cuda")) for k, v in entry["kv"]]
            self.cpu_bytes -= self._kv_bytes(entry["kv"])
        else:  # ssd -> cpu -> hbm
            data = torch.load(entry["file"], map_location="cpu")
            kv = [(k.to("cuda"), v.to("cuda")) for k, v in zip(data["k"], data["v"])]
        entry["kv"] = kv
        entry["tier"] = "hbm"
        self.hbm_bytes += self._kv_bytes(kv)
        self.lru.move_to_end(h)

    def _evict(self, h, target_tier):
        entry = self.blocks[h]
        if target_tier == "cpu" and entry["tier"] == "hbm":
            kv = [(k.to("cpu"), v.to("cpu")) for k, v in entry["kv"]]
            self.hbm_bytes -= self._kv_bytes(entry["kv"])
            entry["kv"] = kv
            entry["tier"] = "cpu"
            self.cpu_bytes += self._kv_bytes(kv)
        elif target_tier == "ssd" and entry["tier"] == "cpu":
            file = os.path.join(self.ssd_dir, f"{h}.pt")
            torch.save({"k": [k for k, _ in entry["kv"]], "v": [v for _, v in entry["kv"]]}, file)
            self.cpu_bytes -= self._kv_bytes(entry["kv"])
            entry["kv"] = None
            entry["tier"] = "ssd"
            entry["file"] = file

    def _find_lru_tier(self, tier):
        if self.eviction_policy == "liveness":
            victim = None
            victim_refs = None
            for h in self.lru:  # oldest first
                if self.blocks[h]["tier"] != tier:
                    continue
                progs = self.block_progs.get(h, {})
                if any(s == "running" for s in progs.values()):
                    continue  # never evict a block an invocation is using
                refs = len(progs)
                if victim is None or refs < victim_refs:
                    victim, victim_refs = h, refs
                    if refs == 0:
                        break  # oldest zero-reference block, nothing better
            return victim
        for h in self.lru:
            if self.blocks[h]["tier"] == tier:
                return h
        return None

    def _enforce_budgets(self):
        while self.hbm_bytes > self.hbm_budget_bytes:
            victim = self._find_lru_tier("hbm")
            if victim is None:
                break
            self._evict(victim, "cpu")
        if self.cpu_budget_bytes is not None:
            while self.cpu_bytes > self.cpu_budget_bytes:
                victim = self._find_lru_tier("cpu")
                if victim is None:
                    break
                self._evict(victim, "ssd")

    def load(self, kv_cache, token_ids, prefix_len):
        if prefix_len <= 0:
            return
        assert prefix_len % self.block_size == 0, "prefix_len must be block-aligned"
        for layer_idx in range(self.num_layers):
            k_chunks = []
            v_chunks = []
            for start in range(0, prefix_len, self.block_size):
                block = token_ids[start:start + self.block_size]
                h = hash_block(block)
                self._ensure_hbm(h)
                k, v = self.blocks[h]["kv"][layer_idx]
                k_chunks.append(k.clone())
                v_chunks.append(v.clone())
            kv_cache.set_layer(layer_idx, torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2))
        self._enforce_budgets()

    def store(self, token_ids, kv_cache, num_prompt_tokens, program_id=None):
        for start in range(0, num_prompt_tokens - num_prompt_tokens % self.block_size, self.block_size):
            block = token_ids[start:start + self.block_size]
            h = hash_block(block)
            if program_id is not None:
                self._mark_running(h, program_id)
            if h in self.blocks:
                self.lru.move_to_end(h)
                continue
            entry = {"token_ids": tuple(block), "tier": "hbm", "kv": [], "file": None}
            for layer_idx in range(self.num_layers):
                k, v = kv_cache.layer_cache[layer_idx]
                entry["kv"].append(
                    (k[:, :, start:start + self.block_size].clone(),
                     v[:, :, start:start + self.block_size].clone())
                )
            self.blocks[h] = entry
            self.lru[h] = None
            self.hbm_bytes += self._kv_bytes(entry["kv"])
        self._enforce_budgets()

    def clear(self):
        self.blocks = {}
        self.lru.clear()
        self.block_progs = {}
        self.prog_blocks = {}
        self.hbm_bytes = 0
        self.cpu_bytes = 0

    def tiers(self):
        counts = {"hbm": 0, "cpu": 0, "ssd": 0}
        for entry in self.blocks.values():
            counts[entry["tier"]] += 1
        return counts
