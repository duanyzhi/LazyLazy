"""从 safetensors 加载模型权重。"""

import glob
import os

from safetensors import safe_open

from lazy.utils.logger import init_logger

logger = init_logger(__name__)


def load_weights(module, model_path: str) -> list[str]:
    """把 safetensors 里的权重按名字一一对应拷进 module。

    最简单的一对一路子：safetensors 里的每个 tensor 名字必须能在
    module.state_dict() 里找到同名参数，否则记进 missing 列表。

    Args:
        module: 目标模型。
        model_path: 存放 *.safetensors 的目录。

    Returns:
        成功加载的参数名列表。
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
                    raise ValueError(f"shape mismatch for {weight_name}: "
                                     f"safetensors={tuple(tensor.shape)}, "
                                     f"param={tuple(param.shape)}")

                param.data.copy_(
                    tensor.to(device=param.device, dtype=param.dtype))
                matched.append(weight_name)
                logger.debug("loaded: %s -> %s", weight_name, file)

    logger.info("matched %d tensors from %s", len(matched), model_path)
    if missing:
        logger.warning("%d tensors not matched:", len(missing))
        for name, file in missing[:10]:
            logger.warning("  %s from %s", name, file)

    return matched
