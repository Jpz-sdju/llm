import torch

# 1. 自动适配硬件环境（公司 CPU / 家里 GPU 自动识别）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ===\n")

# 2. 创建一个需要计算梯度的标量张量 x = 2.0
x = torch.tensor(2.0, requires_grad=True, device=device)

# 3. 定义一个函数 y = x^2 + 3x + 1（前向计算）
y = x**2 + 3 * x + 1

# 4. 自动反向传播求导
y.backward()

# 5. 输出结果
print(f"输入 x 的值: {x.item()}")
print(f"前向计算 y 的值: {y.item()}")
print(f"导数 dy/dx 的值: {x.grad.item()}  (理论导数 2x + 3 = 7.0)")