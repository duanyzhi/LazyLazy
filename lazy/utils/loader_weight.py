import os
import glob

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

