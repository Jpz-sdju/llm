"""模型输入工具：token id → Embedding → ToyLLM。"""

from __future__ import annotations

import torch
import torch.nn as nn

from tokenizer_setup import encode
from toyllm import ToyLLM, init_embedding_, init_linear_


class ToyLLMWithEmbed(nn.Module):
    """Qwen tokenizer 的 id → Embedding → ToyLLM → lm_head，用于 next-token 预测。"""

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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        init_embedding_(self.embed)
        init_linear_(self.lm_head)

    def forward(self, input_ids: torch.Tensor, log_stats: bool = False) -> torch.Tensor:
        """input_ids (B, L) → logits (B, L, vocab_size)；位置 i 预测 token i+1。"""
        x = self.embed(input_ids)
        hidden = self.toyllm(x, log_stats=log_stats)
        return self.lm_head(hidden)


def next_token_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """因果 LM loss：logits[:, i] 与 input_ids[:, i+1] 做交叉熵；pad 位置不计入。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    if attention_mask is None:
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    shift_mask = attention_mask[:, 1:].contiguous().view(-1)
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    return (per_token * shift_mask).sum() / shift_mask.sum()


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
