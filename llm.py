import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ===\n")

n_vectors = 1000  # 交易日数（类似 pandas 10 Minutes to pandas）
width = 1         # 单只股票；>1 时每列独立 cumsum

# 日收益率 ~ N(0, 1)；股价 = 收益率顺次累积（随机游走）
returns = torch.randn(n_vectors, width, device=device)

print(f"【加之前】日收益率，共 {n_vectors} 天，宽 {width}")
print(f"  shape: {tuple(returns.shape)}")
print(f"  Mean: {returns.mean().item():.6f}")
print(f"  Var:  {returns.var().item():.6f}")
show_n = min(8, n_vectors)
for i, v in enumerate(returns[:show_n]):
    print(f"  日 {i:4d}: {[round(t, 4) for t in v.tolist()]}")
if n_vectors > show_n:
    print(f"  ...（共 {n_vectors} 天，略）")

# 股价式累加：沿时间维 cumsum（不是 32 维一次性 sum）
prices = returns.cumsum(dim=0)

print(f"\n【加之后】累积股价（returns.cumsum）")
print(f"  shape: {tuple(prices.shape)}")
print(f"  Mean: {prices.mean().item():.6f}")
print(f"  Var:  {prices.var().item():.6f}")
print(f"  首日/末日: {[round(t, 4) for t in prices[0].tolist()]} → {[round(t, 4) for t in prices[-1].tolist()]}")

plt.figure(figsize=(10, 4))
for j in range(width):
    plt.plot(prices[:, j].cpu().numpy(), linewidth=0.8)
plt.title("Random walk price: cumsum(N(0,1))")
plt.xlabel("day")
plt.ylabel("price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = "llm_price.png"
plt.savefig(out_path, dpi=120)
print(f"\n图已保存: {out_path}")
plt.show()
