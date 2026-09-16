import os
from dataclasses import dataclass

from transformers import AutoConfig

@dataclass(slots=True)
class Config:
    model: str  # model path or hf name
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None

    # prefix cache (task 2: block-granular hash matching)
    prefix_block_size: int = 16

    # weight loading (task 2: "eager" | "three_level" | "two_level")
    weight_loading_mode: str = "eager"
    num_hbm_layers: int | None = None      # decoder layers kept on HBM; None = all
    hbm_budget_bytes: int | None = None    # alternative to num_hbm_layers

    # kv cache manager (task 3: SSD -> CPU -> HBM tiering)
    kv_cache_hbm_budget_bytes: int | None = None    # None = auto from free HBM
    kv_cache_cpu_budget_bytes: int | None = None    # None = unlimited CPU tier
    kv_cache_ssd_dir: str | None = None
    kv_cache_hbm_reserve_bytes: int = 1 << 30       # reserved HBM when auto-sizing


    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        print(f"Loaded config from {self.model}: {self.hf_config}")
        
