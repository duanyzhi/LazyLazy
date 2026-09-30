"""采样器。"""

import torch
from torch import nn


class Sampler(nn.Module):
    """按温度采样下一个 token。"""

    @torch.compile
    def forward(self, logits: torch.Tensor,
                temperatures: torch.Tensor) -> torch.Tensor:
        """从 logits 里采样。

        Args:
            logits: 形状 [batch, vocab] 的原始 logits。
            temperatures: 形状 [batch] 的温度。

        Returns:
            形状 [batch] 的采样 token id。
        """
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(
                dim=-1)
        return sample_tokens
