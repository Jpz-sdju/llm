# ToyLLM

从零搭的迷你 Decoder-only LM（Pre-RMSNorm Attention），用 TinyHelen TinyNews 做 next-token 训练，用来摸清 **Attention / Norm / 数据分布 / batch·step·crop**，不是刷 SOTA。

入口：`make run` → `python3 attention_demo.py`  
可调参数集中在 `attention_demo.py` 顶部「Demo 参数」区。

---

## 快速跑

```bash
make run
# 训练日志 → 项目根目录 log（行缓冲，可实时看）
# 训练结束后终端进入交互续写
```

语料：`data/TinyHelen-zh/TinyNews-zh_000.jsonl`（无则自动从 HF / 镜像下载）。

---

## 今日梳理：训练优化与实验结论

### 1. 精度：FP32 vs FP16 / BF16

| | FP32（当前默认） | FP16 / BF16 |
|--|------------------|-------------|
| 数值稳定性 | 高，少 NaN | FP16 易炸；BF16 相对稳 |
| 工程复杂度 | 无 AMP，直接训 | 需 autocast / GradScaler 等 |
| 显存 / 速度 | 更吃显存、更慢 | 约省一半激活显存，吞吐更高 |
| 适用 | 小模型自学、debug | 工业 pretrain、大模型 |

**结论：** ToyLLM（dim=128、单卡）用 **FP32 合理**。5090 上以后要加速 / 省显存，优先考虑 **BF16 AMP**，不是先上 FP16。

---

### 2. 语料规模 × 轮数：少文多轮 vs 多文少轮

| 策略 | 典型设置 | 现象 |
|------|----------|------|
| **少文 × 多 step** | 2～20 篇，上万～数万 step | loss 极低，**篇首能背熟**；多篇易 **串台**（金价续出佳澜） |
| **多文 × 相对少 step** | 几千～全量，每篇只见几十次 | loss 难到 0，但更像 **学分布**；串台减轻 |

**反直觉点：** 「像正常 LM」要 **多见识、少死记**，不是把几篇文章背穿。  
少文多轮 = 过拟合 / 参数里多篇打架；多文少轮 = 逼模型学通用搭配。

粗算每篇见几次：

\[
\text{每篇平均次数} \approx \frac{\texttt{TRAIN\_STEPS} \times \texttt{BATCH\_SIZE}}{\texttt{CORPUS\_N}}
\]

demo 里可先瞄 **每篇几十次** 量级，不必上千次。

---

### 3. BATCH_SIZE 与 TRAIN_STEPS（别混）

\[
N_{\text{样本}} \approx \texttt{TRAIN\_STEPS} \times \texttt{BATCH\_SIZE}
\]

- `TRAIN_STEPS` = 优化器更新次数  
- `BATCH_SIZE` = **每次**更新吃几篇  

**batch 1→4 却不改 step** ≈ 数据量 ×4（例如 20000×4 = **等价 batch=1 的 80000 篇次**）。

加 batch 的真正用途：

1. **同样本量下** 墙钟更快（GPU 更满）→ step 应近似 ÷batch  
2. 梯度噪声更小  

不要为了「吃满 GPU」只加 batch、不减 step，否则又慢又过训。

---

### 4. 整篇训 vs random crop

| | 整篇 `USE_CROP=False` | crop `USE_CROP=True` |
|--|----------------------|----------------------|
| 训练分布 | 每条序列从 **篇首** 累积 | 窗口可从 **文中任意处** 开始 |
| 交互：篇首 prefix | 通常更好 | 篇首信号被稀释 |
| 交互：中间半句 | 差（训练里少见） | 略好，少文时更易串台 |
| 显存 / 速度 | batch 内 pad 到最长，**L 大 → \(L^2\) 爆炸** | `CROP_MAX` 可 cap \(L\) |

现代 LLM「开头也能续、中间也能续」，靠的是：**海量语料 + 训练大量非篇首窗口**；不是单靠整篇过拟合。  
想两者兼顾：语料加大 + **混训**（一部分整篇、一部分 crop），或推理时中间句 **多贴前文**。

---

### 5. 时间与复杂度（为什么突然变慢）

主导项（Attention）：

\[
t_{\text{step}} \propto B \times L^2 \times D \times N_{\text{layers}}
\]

\[
T_{\text{总}} \approx N_{\text{steps}} \times t_{\text{step}}
\]

\(L\) 从 256→1024，Attention 约 **16×**。  
整篇 + batch pad 到长文时，即使 B=4 也可能顶满 32GB、step 变到秒级。

加速优先序：**cap \(L\)（crop）> 减 step > 减 batch**；再考虑 BF16。

---

### 6. Pre-RMSNorm（已去掉无 Norm 路径）

对照实验结论（详见 `zhihu_no_norm_training.md`）：

- 无 Norm：深层残差激活爆炸 → loss 飙到上千再 NaN  
- Pre-RMSNorm：把每层 Attention 输入钉在 O(1)，训练可走下去  

代码里 **固定 Pre-RMSNorm**，不再跑无 Norm 对照。

---

### 7. Loss 怎么读

- 随机基线 ≈ \(\ln(\text{vocab}) \approx 11.9\)  
- **单篇瞬时 loss** 波动大（随机抽文）；看趋势即可  
- crop 短窗更容易把 loss 压到 <1（背片段）；整篇平均 loss 更高不代表「没学」  
- loss 极低 + 续写重复 / 串台 = **过拟合**，不是成功标准  

交互判据：

- **篇首**能续出训练文风格 → 整篇训有效  
- **中间半句**乱 / 串台 → 分布不匹配或语料太少  

---

### 8. GPU 利用率与显存

| 现象 | 原因 |
|------|------|
| batch=1、短序列，SM ~50% | 算力太碎，CPU encode 拖后腿 |
| batch=8、长 pad，SM 90%+ 但 OOM | 显存顶满，\(B\times L^2\) 过大 |
| 训练很久 log 一直空 | 曾是 stdout **块缓冲**；现已行缓冲 + 终端每 `LOG_EVERY` 打进度 |

建议：先 **BATCH_SIZE=4 + CROP_MAX=512**，盯显存 < 90%，再调大。

---

### 9. 参数怎么配（经验起点）

| 目标 | 建议 |
|------|------|
| 验证能背 1～2 篇 | `CORPUS_N=1～2`，整篇，数千 step，batch=1 |
| 篇首续写 + 少串台 | `CORPUS_N` 加大，整篇或混 crop，**每篇几十次** |
| 中间 prefix 也想试 | `USE_CROP=True`，且语料不要太少 |
| 吃满 5090 又别炸 | batch 4，crop 限制 L；同样本量下 step÷batch |
| 精度 | 先 FP32；再考虑 BF16 |

样本量核对：

\[
N_{\text{样本}} = \texttt{TRAIN\_STEPS} \times \texttt{BATCH\_SIZE}
\quad\text{（改 batch 时记得改 step）}
\]

---

### 10. 文件与入口

| 文件 | 作用 |
|------|------|
| `attention_demo.py` | 唯一训练 / 交互入口，顶栏参数 |
| `toyllm.py` | Pre-RMSNorm Block / Attention |
| `model_input.py` | Embedding、loss、batch pad、续写 |
| `utils.py` | 设备、tokenizer、TinyNews、crop、log 重定向 |
| `zhihu_no_norm_training.md` | 无 Norm 训崩实验笔记 |

权重：当前 demo **训练结束不自动存盘**；需要时用 `torch.save(model.state_dict(), …)`（含各层 W 与 Embedding）。

---

## 待办（原计划）

1. HuggingFace 相关接入 / 用法整理  
2. Weight tying 对比试验（是否像 [这篇讨论](https://www.reddit.com/r/MachineLearning/comments/1meggd2/d_weight_tying_in_llm_seems_to_force_the_last_mlp/) 所说）  

---

## 一句话备忘

> **样本量 = step × batch；少文多轮会背穿串台；多文少轮更像 LM；crop 管中间 prefix 与 \(L^2\)；FP32 稳、BF16 快；Norm 保命。**
