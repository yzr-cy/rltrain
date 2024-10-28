import torch
import torch.nn as nn
from stable_baselines3 import PPO
from zmpEnv import zmpEnv
import os

current_file_path = __file__
absolute_path = os.path.dirname(current_file_path)


# 假设已加载和配置环境
env = zmpEnv()
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[128, 128]),
    activation_fn=nn.ReLU
)

# 创建模型并加载已保存的参数
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
load_path = absolute_path+"/../models/zmpModel.pth"
model.policy.load_state_dict(torch.load(load_path))
model.policy.to("cpu")  # 将模型移动到 CPU
model.policy.eval()     # 切换到评估模式

# 创建一个包装类，以确保 `deterministic=False`
class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super(PolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, x):
        return self.policy(x, deterministic=False)  # 强制使用非确定性模式

# 创建包装后的模型
wrapped_model = PolicyWrapper(model.policy)

# 创建示例输入张量
dummy_input = torch.randn(1, env.observation_space.shape[0]) # type: ignore

# 导出模型为 ONNX 格式
onnx_path = absolute_path+"/../models/zmpModel.onnx"
torch.onnx.export(
    wrapped_model,                # 包装后的模型
    dummy_input,                  # 示例输入张量
    onnx_path,                    # 导出 ONNX 文件的路径
    export_params=True,           # 存储模型参数
    opset_version=11,             # ONNX 的算子版本
    do_constant_folding=True,     # 是否执行常量折叠优化
    input_names=["input"],        # 输入张量名称
    output_names=["output"]       # 输出张量名称
)

print(f"ONNX 模型已导出成功，路径：{onnx_path}")
