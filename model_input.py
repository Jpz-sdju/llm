"""模型输入工具：token id → Embedding → ToyLLM。"""

from __future__ import annotations

import torch
import torch.nn as nn

from tokenizer_setup import encode
from toyllm import ToyLLM, init_embedding_


class ToyLLMWithEmbed(nn.Module):
    """Qwen tokenizer 的 id 经 Embedding 后送入 ToyLLM。"""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        n_layers: int = 8,
        use_norm: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.toyllm = ToyLLM(dim, n_layers, use_norm=use_norm)
        init_embedding_(self.embed)

    def forward(self, input_ids: torch.Tensor, log_stats: bool = False) -> torch.Tensor:
        """input_ids (B, L) → (B, L, dim)"""
        x = self.embed(input_ids)
        return self.toyllm(x, log_stats=log_stats)


def texts_to_input_ids(
    tokenizer,
    texts: str | list[str],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """文本 encode 并 pad → input_ids (B, L), attention_mask (B, L)。单条 str 也走这里。"""
    if isinstance(texts, str):
        texts = [texts]

    batch_ids = [encode(tokenizer, t) for t in texts]
    pad_id = tokenizer.pad_token_id
    max_len = max(len(ids) for ids in batch_ids)

    input_ids = []
    attention_mask = []
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    ids_t = torch.tensor(input_ids, dtype=torch.long)
    mask_t = torch.tensor(attention_mask, dtype=torch.long)
    if device is not None:
        ids_t = ids_t.to(device)
        mask_t = mask_t.to(device)
    return ids_t, mask_t
