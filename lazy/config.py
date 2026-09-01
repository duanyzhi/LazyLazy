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


    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        print(f"Loaded config from {self.model}: {self.hf_config}")
        
