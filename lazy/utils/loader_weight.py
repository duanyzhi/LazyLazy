import os
import glob
from collections import OrderedDict

import torch
from safetensors import safe_open

def load_weights(module, model_path: str):
    """Load safetensors into module parameters with exact name matching.

    This is the simplest one-to-one approach: each tensor name in the safetensors
    files must match a parameter name in module.state_dict().
    """
    print("module: ", module, "\nmodel_path: ", model_path)

    state_dict = module.state_dict()
    matched = []
    missing = []

    for file in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name not in state_dict:
                    missing.append((weight_name, file))
                    continue

                tensor = f.get_tensor(weight_name)
                param = state_dict[weight_name]

                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"shape mismatch for {weight_name}: "
                        f"safetensors={tuple(tensor.shape)}, param={tuple(param.shape)}"
                    )

                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                matched.append(weight_name)
                # print(f"loaded: {weight_name} -> {file}")

    print(f"matched {len(matched)} tensors")
    if missing:
        print("not matched:")
        for name, file in missing[:10]:
            print(f"  {name} from {file}")

    return matched


class WeightLoader:
    """Offloads decoder-layer weights between HBM (cuda), CPU RAM and SSD.

    Modes:
      - "three_level" (SSD -> CPU -> HBM): stage every weight in CPU RAM once, then
        hoist layers to HBM on demand and evict them back to CPU RAM.
      - "two_level" (SSD -> HBM): never stage the full model in RAM. Each layer is
        read from the safetensors files (mmap-backed) straight into HBM and dropped
        to a meta placeholder on eviction. This targets unified-memory devices
        (e.g. nvidia nx) where SSD is the backing store and there is no big CPU
        staging buffer.

    `num_hbm_layers` (or `hbm_budget_bytes`) bounds how many decoder layers stay
    resident on HBM at once; the resident set is managed with LRU eviction.
    embed_tokens + final norm + rotary buffers stay on HBM permanently (small and
    always needed).
    """

    def __init__(self, model, model_path, mode, num_hbm_layers=None, hbm_budget_bytes=None):
        assert mode in ("three_level", "two_level")
        self.model = model
        self.model_path = model_path
        self.mode = mode
        self.num_layers = model.model.config.num_hidden_layers
        self._index_files()
        self._build_groups()
        self.num_hbm_layers = self._resolve_num_hbm_layers(num_hbm_layers, hbm_budget_bytes)
        self._init_resident_and_buffers()
        self.resident = OrderedDict()

    def _index_files(self):
        self.name_to_file = {}
        for file in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
            with safe_open(file, "pt", "cpu") as f:
                for name in f.keys():
                    self.name_to_file[name] = file

    def _build_groups(self):
        self.layer_groups = []  # per decoder layer: [(safetensors name, param)]
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            self.layer_groups.append(
                [(f"model.layers.{i}.{n}", p) for n, p in layer.named_parameters()]
            )
        self.resident_params = [
            ("model.embed_tokens.weight", self.model.model.embed_tokens.weight),
            ("model.norm.weight", self.model.model.norm.weight),
        ]

    def _resolve_num_hbm_layers(self, num_hbm_layers, hbm_budget_bytes):
        if num_hbm_layers is not None:
            n = int(num_hbm_layers)
        elif hbm_budget_bytes is not None:
            resident_bytes = sum(p.numel() * p.element_size() for _, p in self.resident_params)
            layer_bytes = sum(p.numel() * p.element_size() for _, p in self.layer_groups[0])
            n = int((hbm_budget_bytes - resident_bytes) // layer_bytes)
        else:
            n = self.num_layers
        return max(1, min(n, self.num_layers))

    def _init_resident_and_buffers(self):
        if self.mode == "three_level":
            load_weights(self.model, self.model_path)  # SSD -> CPU RAM (full model)
        else:
            for _, param in self.model.named_parameters():
                self._drop_param(param)

        # rotary cos/sin buffers stay on HBM permanently: they are tiny, stateless,
        # and shared across layers, so offloading whole layers would lose their data.
        for buf in self.model.buffers():
            buf.data = buf.data.to("cuda")

        for name, param in self.resident_params:
            if self.mode == "three_level":
                param.data = param.data.to("cuda")
            else:
                param.data = self._read_tensor(name).to("cuda", dtype=param.dtype)

    @staticmethod
    def _drop_param(param):
        # A zero-element tensor frees the backing storage while keeping dtype;
        # shape is restored from the safetensors file on the next load.
        param.data = torch.empty(0, dtype=param.dtype, device="cpu")

    def _read_tensor(self, name):
        with safe_open(self.name_to_file[name], "pt", "cpu") as f:
            return f.get_tensor(name)

    def _load_layer(self, i):
        for name, param in self.layer_groups[i]:
            if self.mode == "three_level":
                param.data = param.data.to("cuda")
            else:
                param.data = self._read_tensor(name).to("cuda", dtype=param.dtype)

    def _unload_layer(self, i):
        for name, param in self.layer_groups[i]:
            if self.mode == "three_level":
                param.data = param.data.to("cpu")
            else:
                self._drop_param(param)

    def ensure_layer(self, i):
        if i in self.resident:
            self.resident.move_to_end(i)
            return
        self._load_layer(i)
        self.resident[i] = None
        while len(self.resident) > self.num_hbm_layers:
            oldest, _ = self.resident.popitem(last=False)
            self._unload_layer(oldest)

    @property
    def num_resident(self):
        return len(self.resident)

