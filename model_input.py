"""模型输入工具：token id → Embedding → ToyLLM。"""

from __future__ import annotations

import torch
import torch.nn as nn

from tokenizer_setup import decode, encode
from toyllm import ToyLLM, init_embedding_


class ToyLLMWithEmbed(nn.Module):
    """Qwen tokenizer 的 id → Embedding → ToyLLM → lm_head（weight tying），用于 next-token 预测。"""

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
        # Weight tying：输出头与词嵌入共享同一份 weight
        self.lm_head.weight = self.embed.weight

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


@torch.no_grad()
def greedy_continue(
    model: ToyLLMWithEmbed,
    tokenizer,
    prompt: str,
    n_tokens: int,
    *,
    device: torch.device,
) -> tuple[list[int], str]:
    """给定 prompt，贪心续写 n_tokens 个 token。返回 (新 token ids, 解码文本)。"""
    model.eval()
    ids = encode(tokenizer, prompt)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    for _ in range(n_tokens):
        logits = model(cur)
        next_id = int(logits[0, -1].argmax().item())
        new_ids.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
    return new_ids, decode(tokenizer, new_ids)


def interactive_ask(
    model_norm: ToyLLMWithEmbed,
    model_nonorm: ToyLLMWithEmbed,
    tokenizer,
    *,
    device: torch.device,
    max_new_tokens: int = 32,
) -> None:
    """训练结束后：终端输入前缀，两个模型贪心续写对比。"""
    print(f"\n{'=' * 75}")
    print("交互问答：输入前缀，看模型怎么续写（空行或 q 退出）")
    print(f"续写长度: {max_new_tokens} tokens")
    print("提示: 空行或 q 退出；训练 log 已写入项目 log 文件")
    print("=" * 75)
    while True:
        try:
            prompt = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break
        if not prompt or prompt.lower() in {"q", "quit", "exit"}:
            print("[退出]")
            break
        _, cont_n = greedy_continue(
            model_norm, tokenizer, prompt, max_new_tokens, device=device
        )
        _, cont_0 = greedy_continue(
            model_nonorm, tokenizer, prompt, max_new_tokens, device=device
        )
        print(f"有 Norm → {prompt}{cont_n}")
        print(f"无 Norm → {prompt}{cont_0}")
