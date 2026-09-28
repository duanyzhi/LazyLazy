import os
import glob

from safetensors import safe_open

from lazy.utils.logger import init_logger

logger = init_logger(__name__)


def load_weights(module, model_path: str):
    """Load safetensors into module parameters with exact name matching.

    This is the simplest one-to-one approach: each tensor name in the safetensors
    files must match a parameter name in module.state_dict().
    """
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
                logger.debug("loaded: %s -> %s", weight_name, file)

    logger.info("matched %d tensors from %s", len(matched), model_path)
    if missing:
        logger.warning("%d tensors not matched:", len(missing))
        for name, file in missing[:10]:
            logger.warning("  %s from %s", name, file)

    return matched
