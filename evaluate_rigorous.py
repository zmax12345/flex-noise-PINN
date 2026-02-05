import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SpeckleFlowDataset
from model import SpecklePINN
import os

# ================= 配置 =================
CONFIG = {
    # 🔥 1. 这里必须改成你新数据的路径，并且 Key 要和 dataset.py 里的 elif 对应！
    'roots': {
        # 例如：你的 dataset.py 里写的是 elif 'new_experiment' in group_name...
        'group_2.3': '/data/zm/2026.1.12_testdata/2.3/'
    },

    'window_size_us': 150000,
    'step_size_us': 50000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 指向你刚才用全量数据训练出来的那个新模型
    'model_path': '/data/zm/2026.1.12_testdata/2.5PINN_Result/model_train/best_model.pth',

    # 🔥 2. 这里设为空列表 []
    # 意思是："不要过滤，把新文件夹里的所有流速文件都测一遍"
    # 如果你只填 [0.8, 1.8], 它就只加载这两个流速的文件，其他的会跳过。
    'holdout_flows': []
}


def evaluate_rigorous():
    print("Loading EVALUATION dataset (New Data)...")

    # 3. 这里 mode='train' 还是 'val' 都可以，因为 holdout_flows 是空的
    # 但为了逻辑一致，既然是做验证，用 mode='val' 且 holdout=[] (全不保留=全都要)
    # 等等，dataset.py 里的逻辑是：
    # if mode == 'val' and not is_holdout: continue
    # 如果 holdout_flows 为空，is_holdout 永远是 False，那 'val' 模式下什么都读不到！

    # 🔥 必须用 mode='train' 配合 holdout_flows=[]
    # 才能骗过 dataset.py 读取所有文件
    val_ds = SpeckleFlowDataset(CONFIG['roots'], mode='train',
                                holdout_flows=CONFIG['holdout_flows'],
                                window_size_us=CONFIG['window_size_us'],
                                step_size_us=CONFIG['step_size_us'])

    # 不打乱，按顺序取
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = SpecklePINN().to(CONFIG['device'])
    if not os.path.exists(CONFIG['model_path']):
        print(f"Model not found at {CONFIG['model_path']}")
        return
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    results = {}

    print("Running Inference on New Data...")
    with torch.no_grad():
        for batch in val_loader:
            g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
            aux = batch['aux_input'].to(CONFIG['device']).float()
            v_label = batch['flow_label'].item()
            m_val = batch['k_factor'].to(CONFIG['device']).float()

            out = model(g2_obs, aux, m_val)
            v_pred = out['v_pred'].item()
            g2_hat = out['g2_hat'].cpu().numpy()[0]
            g2_obs = g2_obs.cpu().numpy()[0]

            if v_label not in results:
                results[v_label] = {'preds': [], 'errs': [], 'curves': []}

            results[v_label]['preds'].append(v_pred)
            results[v_label]['errs'].append(abs(v_pred - v_label))
            # 存几个曲线画图用
            if len(results[v_label]['curves']) < 2:
                results[v_label]['curves'].append((g2_obs, g2_hat, v_pred))

    # --- 绘图与统计 ---
    unique_flows = sorted(results.keys())
    if len(unique_flows) == 0:
        print("No samples found! Check dataset path and keys.")
        return

    # 动态调整画布大小
    fig, axes = plt.subplots(len(unique_flows), 2, figsize=(12, 4 * len(unique_flows)))
    if len(unique_flows) == 1: axes = axes.reshape(1, -1)

    print("\n========= 新数据泛化测试报告 =========")

    for i, flow in enumerate(unique_flows):
        data = results[flow]
        mean_mae = np.mean(data['errs'])
        mean_pred = np.mean(data['preds'])
        std_pred = np.std(data['preds'])

        print(f"流速: {flow:.2f} mm/s")
        print(f"   -> 平均预测: {mean_pred:.2f} ± {std_pred:.2f}")
        print(f"   -> MAE: {mean_mae:.4f}")
        if flow != 0:
            print(f"   -> 相对误差: {(mean_mae / flow) * 100:.2f}%")

        # 画左图：误差分布散点
        ax_scatter = axes[i, 0] if len(unique_flows) > 1 else axes[0]
        ax_scatter.hist(data['preds'], bins=20, alpha=0.7, color='green', label='Preds')
        ax_scatter.axvline(flow, color='red', linestyle='--', linewidth=2, label='Ground Truth')
        ax_scatter.set_title(f"Label v={flow:.2f} | MAE={mean_mae:.2f}")
        ax_scatter.legend()

        # 画右图：曲线拟合情况
        ax_curve = axes[i, 1] if len(unique_flows) > 1 else axes[1]
        if len(data['curves']) > 0:
            obs, hat, pred_v = data['curves'][0]
            ax_curve.plot(obs, 'b.', alpha=0.5, label='Observed')
            ax_curve.plot(hat, 'r-', linewidth=2, label=f'PINN (v={pred_v:.2f})')
            ax_curve.set_title(f"Curve Fitting (Sample)")
            ax_curve.legend()
            ax_curve.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/data/zm/2026.1.12_testdata/2.3/generalization_test_result.png')
    print("====================================")
    print("结果图已保存至 generalization_test_result.png")


if __name__ == "__main__":
    evaluate_rigorous()