# 亲手训崩了一个 16 层 Transformer：无 Norm 时 loss 为什么会飙到 NaN？

先说背景：我在用 **ToyLLM** 从零搭一个迷你语言模型——16 层 Pre-RMSNorm Transformer、Qwen tokenizer、TinyNews 约 4 万篇中文新闻做 next-token 训练。目的不是刷 SOTA，而是**把 Attention、Norm、训练稳定性这几件事摸透**。

一开始我做「有 Norm vs 无 Norm」对照实验：同样初始化、同样学习率、同样语料，只关掉 Attention 前的 RMSNorm。结果非常干脆——**无 Norm 在 step 500 loss 就上千，step 1000 直接 NaN**；有 Norm 则 loss 从 11.9 稳降到 7～8 附近。

这篇回答把这次「训崩」的过程拆开讲：**不是语料坏了，而是深层残差网络没有 Norm 拴住激活尺度，训练必然数值爆炸**。

主要讲 4 块：

1. 实验设置与 log 里发生了什么
2. 为什么一开始两个模型 loss 差不多
3. 无 Norm 是怎么一步步炸掉的（残差放大链）
4. 有 Norm 为什么能撑住——以及现代 LLM 为什么几乎离不开 Norm

---

## 一、实验设置：对照组只差一个开关

我的 Block 结构是典型的 Pre-Norm 残差：

```
x → RMSNorm → Attention → 残差加回 x
x → FFN → 再残差加回 x
```

对照组把 Attention 入口**跳过 RMSNorm**（当时代码里用 `use_norm=False` 开关），其余完全一致：

| 配置 | 值 |
|------|-----|
| 层数 | 16 |
| 隐藏维度 | 128 |
| 优化器 | Adam，lr = 1e-3 |
| 训练步数 | 3000（每 step 从语料随机抽 1 篇新闻） |
| 语料 | TinyHelen TinyNews，约 39984 篇 |
| 初始化 | Embedding / Linear 均 N(0, 0.02²)，与 GPT-2 同量级 |

**关键判据**：除了 loss 能否下降，我还盯末层隐藏状态的 **X.rms**（均方根），看它有没有失控放大。

---

## 二、log 记录：500 step 内从「差不多」到「崩盘」

下面是我真实跑出来的数字（CUDA，同 seed 初始化）：

| Step | 有 Norm loss | 无 Norm loss | 无 Norm 末层 X.rms |
|------|-------------|-------------|-------------------|
| 0 | 11.93 | 11.93 | **0.023** |
| 500 | 7.67 | **1323** | **8249** |
| 1000 | 8.11 | **NaN** | NaN |

几个值得注意的点：

**1. Step 0 几乎一样**

随机初始化后，两个模型的 loss 都在 $\ln(151669) \approx 11.9$ 附近——这就是「词表均匀猜」的交叉熵基线。此时无 Norm 的 X.rms 甚至更小（0.023），**看起来「更稳」**。

**2. Step 500：loss 不是「稍微变差」，是三个数量级的失控**

无 Norm loss 1323，同时 X.rms 从 0.02 飙到 **8249**。这不是「学不好」，是**网络内部数值已经炸了**。

**3. Step 1000：NaN**

CrossEntropy 在极大 logits 下，`softmax` 和 `log` 会出现 `inf - inf`，梯度也变 NaN，权重一步就全坏。

> 有 Norm 在同配置下 X.rms 只到 ~1.7（step 500）、~3.6（step 1000），loss 平稳下降。**差别不在语料，而在每层有没有把激活尺度拉回 O(1)。**

---

## 三、无 Norm 为什么不行：16 层残差是一条「越滚越大」的链

### 3.1 残差堆叠 = 激活可以无限长大

每个 Block 做两次残差：

$$
x \leftarrow x + \mathrm{Attn}(x), \qquad x \leftarrow x + \mathrm{FFN}(x)
$$

无 Norm 时，Attention 的 Q/K/V **直接吃 raw 的 x**。若某一层的 $\|x\|$ 偏大：

- Q、K、V 线性投影后偏大 → Attention 输出偏大
- 残差 `$x + \cdots$` 把大向量**叠**在原来的 x 上 → 下一层输入更大
- 16 层重复 → **指数式放大**（我 log 里 500 step 就到 8000+ RMS）

有 Pre-RMSNorm 时，Attention 前先做：

$$
\hat{x} = \frac{x}{\mathrm{RMS}(x)}
$$

每层 Attention 看到的输入 RMS **被钉在 ~1**，这条放大链在每一层入口都被掐断。

### 3.2 FFN 也在推波助澜

我的 FFN 是 `dim → 4×dim → dim`，中间 GELU，**整条 FFN 路径也没有 Norm**。无 Norm 时，FFN 同样会放大信号，和 Attention 残差**叠加**放大。

现代 LLM（LLaMA、Qwen 等）普遍是 **Pre-RMSNorm + 残差**，有的还在 FFN 前再加 Norm——核心目的都是同一个：**别让激活尺度随深度漂移**。

### 3.3 从 X.rms 爆炸到 loss = NaN 的路径

训练目标是 next-token 交叉熵：末层 hidden 经 `lm_head`（与 Embedding 共享权重）得到 logits，再对正确 token 做 $-\log p$。

当 hidden RMS 到 8000+ 时：

1. logits 某些维度可达 **上千、上万**
2. `softmax(logits)` 在 float32 下极端值 → 上溢/下溢
3. CrossEntropy 反向传播出现 **Inf/NaN 梯度**
4. Adam 一步更新后，权重含 NaN → 之后 forward 全 NaN

所以用户看到的不是「loss 缓慢上升到正无穷」，而是 **先飙到一个荒谬的大数（1323），再突然 NaN**。

### 3.4 别被 A.max = 1.0 骗了

我 log 里 Attention 权重 **A.max 一直 ≈ 1.0**，无 Norm 崩了也如此。

原因很简单：Softmax 每行和恒为 1，**最大值当然可以是 1**。这只能说明「归一化算对了」，**不能**说明 Attention 在学有意义的模式。

真正该看的是 **进入 Attention 之前的 x 有多大**（X.rms），以及 **Scores 的 spread**（无 Norm 崩后 S.std 也变 NaN）。

---

## 四、为什么单条短文「好像能训」，换多文档就崩？

我 earlier 用**一条固定 MOD 说明文字**过拟合时，无 Norm 有时还能勉强降 loss；换成 **TinyNews 每 step 随机抽一篇** 后，无 Norm 很快炸。

原因并不神秘：

| 单条过拟合 | 多文档随机训练 |
|-----------|---------------|
| 同一段文本反复见，梯度方向一致 | 每 step 不同文章，梯度方向乱跳 |
| 序列短（几十 token） | 中位 ~226 token，更长 |
| 模型可以「背」住一条样本 | 必须学泛化，激活尺度更敏感 |

**多文档训练更接近真实 LM 场景**，对数值稳定性要求更高；无 Norm 在这种设置下几乎必崩。单条过拟合能撑住，**不能**说明「深层 Transformer 可以不要 Norm」——那只是问题太简单、隐藏了不稳定性。

---

## 五、有 Norm 在干什么（一句话版）

Pre-RMSNorm 的作用不是「让 loss  magically 更低」，而是：

> **在每一层 Attention（以及很多架构里的 FFN）之前，把输入向量的 RMS 拉回 O(1)，打断残差堆叠里的尺度正反馈。**

在这个 O(1) 的尺度上：

- Q/K 点积后的 Scores 方差可控（再配合 `/√d`）
- Softmax 不会一上来就极端 one-hot 或数值溢出
- 反向传播梯度不会在 16 层里指数放大

我另一篇回答从**方差推导**讲过 Embedding → RMSNorm → Q/K → `/√d` → Softmax 整条链；这篇是从**训练实验**补上了「没有 Norm 链会断在哪里」。

---

## 六、能不能「救活」无 Norm？

如果坚持去掉 RMSNorm，只能降低不稳定因素，**不能指望和 Norm 版同等深度同等 lr 稳定训练**：

- 减少层数（16 → 4）
- 降低学习率（1e-3 → 1e-4）
- 梯度裁剪 `clip_grad_norm_`
- 更小的初始化 std

但这些是**绕开**问题，不是替代 Norm。工业界 几十层～上百层 的 Decoder-only 模型，**Norm 是标配**，不是没有历史原因。

---

**总结一下：**

1. **无 Norm 训深层 Transformer，loss 爆炸是预期现象**：残差堆叠让激活 RMS 指数放大，CrossEntropy 数值溢出变 NaN。
2. **Step 0 loss 一样说明不了问题**；要看训练过程中 **X.rms 是否失控**。
3. **A.max = 1 不是 Attention 正常的证据**；Softmax 行和恒为 1，掩盖了内部尺度已经崩坏。
4. **Pre-RMSNorm 的价值**：每层把激活钉在 O(1)，让 16 层 + 多文档 + lr=1e-3 这种「像样」的训练配置跑得下去。

以上是我 ToyLLM 对照实验的一手记录。你在复现类似小模型时，有没有也遇到过「无 Norm 前几步看着还行，几百 step 突然 NaN」？欢迎评论区贴 log 对比。

如果觉得这种「从零搭、拿 log 说话」的笔记有用，点个赞让更多人看到~

---

**话题标签建议：** #Transformer #深度学习 #大语言模型 #RMSNorm #神经网络训练

**内容自检清单：**

- [x] 标题体现专业视角（亲手实验 + 具体问题）
- [x] 开头建立可信度（ToyLLM 一手实验）
- [x] 论点有 log 数据支撑
- [x] 机制解释与实验现象对应
- [x] 避免广告式语言
- [x] 结尾有互动引导
