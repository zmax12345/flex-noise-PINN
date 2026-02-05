import torch
import torch.nn as nn
import numpy as np

# Grid 转为 秒 (保持不变)
TAU_LAGS_US = np.unique(np.concatenate([
    # 0 ~ 0.5 ms : 10 us step
    np.arange(0, 500, 10),
    # 0.5 ~ 5 ms : 100 us step
    np.arange(500, 5001, 100),
    # 5 ~ 100 ms : 1 ms step
    np.arange(5000, 150001, 1000),
])).astype(np.float32)
TAU_GRID_SECONDS = TAU_LAGS_US * 1e-6


class SpecklePINN(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=128):
        super().__init__()
        # 注册时间轴 buffer
        self.register_buffer('tau_grid', torch.tensor(TAU_GRID_SECONDS))

        if input_dim is None:
            input_dim = int(self.tau_grid.numel())

        # 主干网络
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )

        # === 核心物理参数头 ===
        self.head_tau = nn.Linear(hidden_dim // 2, 1)   # 流速衰减时间
        self.head_beta = nn.Linear(hidden_dim // 2, 1)  # 仪器反差因子
        self.head_alpha = nn.Linear(hidden_dim // 2, 1) # 衰减指数 (1.0~2.0)

        # === 🔥 新增：混合模型参数头 (修复报错的关键) ===
        # 1. 静态散射占比 rho (0~1)
        self.head_rho = nn.Linear(hidden_dim // 2, 1)
        
        # 2. 132Hz 结构化噪声建模
        # 振幅 A 和 相位 phi
        self.head_noise_amp = nn.Linear(hidden_dim // 2, 1)
        self.head_noise_phi = nn.Linear(hidden_dim // 2, 1)

        # 初始化 bias，让训练初期更稳定
        nn.init.constant_(self.head_tau.bias, 0.0)
        nn.init.constant_(self.head_rho.bias, -2.0) # 初始 rho 偏小 (sigmoid后约0.1)

    def forward(self, g2_curve, aux_input, m_value):
        x = torch.cat([g2_curve, aux_input], dim=1)
        feat = self.backbone(x)

        # 1. 预测物理参数
        # Tau_c: 限制在合理范围 [1us, 100ms]
        tau_c = torch.sigmoid(self.head_tau(feat)) * (0.1 - 1e-6) + 1e-6

        # Beta: 0~1
        beta = torch.sigmoid(self.head_beta(feat))
        
        # Alpha: 1.0 (布朗) ~ 2.0 (定向流)
        alpha = torch.sigmoid(self.head_alpha(feat)) + 1.0

        # 🔥 Rho: 动态光占比 (0~1)
        rho = torch.sigmoid(self.head_rho(feat))

        # 🔥 Noise: 132Hz 噪声参数
        noise_amp = torch.sigmoid(self.head_noise_amp(feat)) * 0.2 # 限制最大振幅 0.2
        noise_phi = torch.sigmoid(self.head_noise_phi(feat)) * 2 * np.pi

        # --- 2. 物理模型生成 (Mixed Model) ---
        t = self.tau_grid.unsqueeze(0) + 1e-9 # [1, N_lags]

        # 动态部分 g1 (High Frequency)
        term = t / tau_c
        exponent = -2.0 * (term ** alpha)
        exponent = torch.clamp(exponent, min=-20.0, max=0.0) # 防止数值不稳定
        g1_dynamic = torch.exp(exponent / 2.0) # 注意: Siegert关系里是 |g1|^2，这里先算 g1

        # 混合场 g1 (Heterodyne mixing)
        # static part = 1.0
        g1_total = rho * g1_dynamic + (1.0 - rho)

        # 基础物理项 g2
        g2_physics = 1.0 + beta * (g1_total ** 2)

        # --- 3. 添加 132Hz 周期噪声 ---
        omega = 2 * np.pi * 132.0
        noise_term = noise_amp * torch.cos(omega * t + noise_phi)

        # 最终重构的 g2
        g2_hat = g2_physics + noise_term

        # --- 4. 流速预测 ---
        # v = m / tau_c
        v_pred = m_value / tau_c

        return {
            'tau_c': tau_c,
            'beta': beta,
            'alpha': alpha,
            'rho': rho,         # 返回 rho 供分析
            'g2_hat': g2_hat,
            'v_pred': v_pred
        }
