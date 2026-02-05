import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from dataset import SpeckleFlowDataset
from model import SpecklePINN

# ================= 配置 =================
CONFIG = {
    # 1. 训练集路径 (原本的那两组)
    'train_roots': {
        'group_680W': '/data/zm/2026.1.12_testdata/1.15_150_680W/',
        'group_gaoyuzhi': '/data/zm/2026.1.12_testdata/gaoyuzhi/'
    },

    # 2. 验证集路径 (你的新数据放在这里!)
    # 请修改为你新数据的真实路径
    'val_roots': {
        'group_580': '/data/zm/2026.1.12_testdata/1.15_150_580W/'
    },

    'window_size_us': 400000,
    'step_size_us': 50000,
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lambda_flow': 1.0,
    'lambda_fit': 10.0,  # 如果用了加权，这个可以适当降低，比如 1.0
    'save_dir': '/data/zm/2026.1.12_testdata/2.5PINN_Result/model_train',  # 换个文件夹存权重，别覆盖了之前的
}


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # --- 1. 加载训练集 (全量) ---
    print("Loading TRAIN dataset (All Old Data)...")
    # holdout_flows 设为空 []，表示不保留，全部用于训练
    train_ds = SpeckleFlowDataset(
        data_roots=CONFIG['train_roots'],
        mode='train',
        holdout_flows=[],  # <--- 关键修改：空列表 = 全量训练
        window_size_us=CONFIG['window_size_us'],
        step_size_us=CONFIG['step_size_us']
    )

    # --- 2. 加载验证集 (新数据) ---
    print("Loading VAL dataset (New Unseen Data)...")
    # 这里的 mode 无所谓了，因为 holdout_flows 为空，
    # 但为了逻辑通顺，我们还是设为 'val'，且 holdout 设为空（表示该文件夹下所有文件都是验证集）
    # 注意：dataset.py 的逻辑是：
    # if mode='val' and not is_holdout: continue
    # 所以为了让它读取所有新文件，我们需要一个小技巧：
    # 把 holdout_flows 设为 None 或者一个特殊标记？
    # 不，最简单的办法是直接改一下 dataset.py 让它更灵活，
    # 或者直接用 mode='train' (因为 train 模式下如果不匹配 holdout 就会读取)，
    # 但这听起来很怪。

    # 💡 最优解：稍微改一下 dataset.py 的逻辑，或者简单粗暴地：
    # 在下面调用时，把 mode='train' 传给验证集 (意思是"读取所有非排除文件")
    # 因为我们的 val_roots 里全是新数据，我们希望全读进来，且没有任何排除项。
    val_ds = SpeckleFlowDataset(
        data_roots=CONFIG['val_roots'],
        mode='train',  # 这里用 'train' 是为了骗过 dataset.py 让它读取所有文件
        holdout_flows=[],  # 不排除任何文件
        window_size_us=CONFIG['window_size_us'],
        step_size_us=CONFIG['step_size_us']
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    print(f"Data split: Train={len(train_ds)} slices, Val (New)={len(val_ds)} slices")

    # 2. 模型
    model = SpecklePINN().to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 3. 训练
    print("Start Training (Rigorous Physics Mode)...")
    history = {'train_loss': [], 'val_loss': []}

    # 定义 Fit Loss 的权重 (可选：给头部更高权重)
    # 既然归一化修好了，暂时用均匀权重
    # Fit loss 权重：强调早期下降段（你关心的前 1ms / 5ms）
    tau_us = (model.tau_grid.detach().cpu().numpy() * 1e6).astype(np.float32)
    w = np.ones_like(tau_us, dtype=np.float32)
    w[tau_us <= 1000.0] = 5.0
    w[(tau_us > 1000.0) & (tau_us <= 5000.0)] = 2.0
    w[tau_us > 100000.0] = 1.5  # 🔥 给 100ms 以后稍微加一点点权重，强迫模型看慢速衰减
    # 归一化：让平均权重为 1，避免等效 lambda_fit 突变
    w = w / (np.mean(w) + 1e-9)
    fit_weights = torch.from_numpy(w).to(CONFIG['device'])

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", unit="batch")

        for batch in pbar:
            g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
            aux = batch['aux_input'].to(CONFIG['device']).float()
            v_label = batch['flow_label'].to(CONFIG['device']).float()
            m_val = batch['k_factor'].to(CONFIG['device']).float()

            optimizer.zero_grad()

            out = model(g2_obs, aux, m_val)

            # Loss 计算
            g2_hat = out['g2_hat']

            # Fit Loss
            loss_fit = torch.mean(fit_weights * (g2_hat - g2_obs) ** 2)

            # Flow Loss
            v_pred = out['v_pred']
            loss_flow = torch.mean((v_pred - v_label) ** 2)

            loss = CONFIG['lambda_fit'] * loss_fit + CONFIG['lambda_flow'] * loss_flow

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            pbar.set_postfix({
                'L': f"{loss.item():.2f}",
                'Fit': f"{loss_fit.item():.2f}",
                'Flow': f"{loss_flow.item():.2f}"
            })

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        history['train_loss'].append(avg_loss)

        # === 验证 ===
        model.eval()
        val_loss_sum = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
                aux = batch['aux_input'].to(CONFIG['device']).float()
                v_label = batch['flow_label'].to(CONFIG['device']).float()
                m_val = batch['k_factor'].to(CONFIG['device']).float()

                out = model(g2_obs, aux, m_val)
                v_err = torch.abs(out['v_pred'] - v_label).mean()

                val_loss_sum += v_err.item()
                val_count += 1

        avg_val_mae = val_loss_sum / val_count if val_count > 0 else 0.0
        history['val_loss'].append(avg_val_mae)

        scheduler.step(avg_val_mae)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val MAE (Unseen Flows): {avg_val_mae:.4f}")

        if epoch > 0 and avg_val_mae < min(history['val_loss'][:-1]):
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val MAE (Holdout)')
    plt.legend()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'training_result.png'))
    print("Rigorous Training Complete.")


if __name__ == "__main__":
    main()