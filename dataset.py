import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import correlate
from tqdm import tqdm  # 用于显示预处理进度

# 96点 Grid
TAU_LAGS = np.unique(np.concatenate([
    np.arange(0, 500, 10),
    np.arange(500, 5001, 100),
    np.arange(5000, 150001, 1000),
])).astype(np.int64)


class SpeckleFlowDataset(Dataset):
    def __init__(self, data_roots, mode='train', holdout_flows=None,
                 window_size_us=150000, step_size_us=50000,
                 patch_size=64, stride=16):
        self.window_size_us = int(window_size_us)
        self.step_size_us = int(step_size_us)
        self.patch_size = patch_size
        self.stride = stride
        self.tau_lags = TAU_LAGS

        # 缓存列表
        self.processed_samples = []  # 直接存处理好的 Tensor
        self.mode = mode
        self.holdout_flows = holdout_flows if holdout_flows is not None else []
        self.dt_us = 10

        print(f"Dataset ({mode}) initializing with Pre-computation...")
        print(f"   -> Patch Size: {self.patch_size}x{self.patch_size}, Stride: {self.stride}")

        self._load_and_process_all(data_roots)

        print(f"Dataset ({mode}) ready: {len(self.processed_samples)} pre-computed samples.")

    def _load_and_process_all(self, roots):
        # 临时存储待处理的任务，稍后统一处理
        raw_tasks = []

        # 1. 扫描文件并生成 Patch 任务
        for group_name, root_dir in roots.items():
            if not os.path.exists(root_dir): continue

            # 匹配物理参数 m
            if 'gaoyuzhi' in group_name:
                current_m = 0.014611
            elif 'group_680W' in group_name:
                current_m = 0.0105
            elif 'group_580' in group_name:
                current_m = 0.0114853
            elif 'group_122' in group_name:
                current_m = 0.010154
            elif 'group_pianzhen1' in group_name:
                current_m = 0.010157
            elif 'group_2.3' in group_name:
                current_m = 0.010099
            elif 'group_2.4' in group_name:
                current_m = 0.00996
            else:
                current_m = 0.011167

            print(f"   -> Scanning '{group_name}'...")
            files = glob.glob(os.path.join(root_dir, "*.csv"))

            for fpath in files:
                try:
                    fname = os.path.basename(fpath)
                    try:
                        name_clean = fname.replace("_clip.csv", "").replace("mm.csv", "").replace("mm", "")
                        flow_val = float(name_clean)
                    except:
                        continue

                    is_holdout = False
                    for hv in self.holdout_flows:
                        if abs(flow_val - hv) < 0.01:
                            is_holdout = True
                            break
                    if self.mode == 'train' and is_holdout: continue
                    if self.mode == 'val' and not is_holdout: continue

                    # 读取数据
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        df = pd.read_csv(f, header=None, usecols=[0, 1, 2], dtype=str, engine='c', on_bad_lines='skip')
                    df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(np.int64)

                    # 固定 ROI 过滤 (400-500)
                    df = df[(df.iloc[:, 0] >= 400) & (df.iloc[:, 0] < 500)]
                    df = df[df.iloc[:, 1] <= 768]

                    if len(df) < 1000: continue
                    data_np = df.values

                    # 固定范围切分
                    FIXED_ROW_START, FIXED_ROW_END = 400, 500
                    FIXED_COL_START, FIXED_COL_END = 0, 768

                    for r in range(FIXED_ROW_START, FIXED_ROW_END, self.stride):
                        for c in range(FIXED_COL_START, FIXED_COL_END, self.stride):
                            r_end = r + self.patch_size
                            c_end = c + self.patch_size

                            mask = (data_np[:, 0] >= r) & (data_np[:, 0] < r_end) & \
                                   (data_np[:, 1] >= c) & (data_np[:, 1] < c_end)
                            patch_events = data_np[mask]

                            # 阈值提高到 500
                            if len(patch_events) < 500: continue

                            tin_array = np.ascontiguousarray(np.sort(patch_events[:, 2]))
                            duration = tin_array[-1] - tin_array[0]
                            if duration <= self.window_size_us: continue

                            # 添加到待处理列表
                            raw_tasks.append({
                                'tin': tin_array,
                                'label': flow_val,
                                'm': current_m
                            })

                except Exception as e:
                    print(f"Skip {fpath}: {e}")

        # 2. 统一进行预计算 (Pre-compute g2)
        print(f"   -> Pre-computing g2 for {len(raw_tasks)} patches... (This may take a minute)")

        for task in tqdm(raw_tasks, desc="Processing G2"):
            self._process_single_trace(task['tin'], task['label'], task['m'])

    def _process_single_trace(self, t_all, label, m_val):
        t_min, t_max = t_all[0], t_all[-1]
        start_times = np.arange(t_min, t_max - self.window_size_us + 1, self.step_size_us)

        idx_starts = np.searchsorted(t_all, start_times)
        idx_ends = np.searchsorted(t_all, start_times + self.window_size_us)
        counts = idx_ends - idx_starts

        # 再次过滤，确保切片内事件足够
        valid_indices = np.where(counts > 500)[0]

        for i in valid_indices:
            start_idx = idx_starts[i]
            end_idx = idx_ends[i]

            # 取出时间片段
            ts = t_all[start_idx:end_idx]
            ts = ts - ts[0]  # 归零

            # === 计算 g2 (最耗时步骤，现在只做一次) ===
            num_bins = self.window_size_us // self.dt_us
            I_t, _ = np.histogram(ts, bins=num_bins, range=(0, self.window_size_us))
            I_t = I_t.astype(np.float32)

            acf = correlate(I_t, I_t, mode='full')
            center = len(acf) // 2
            acf_right = acf[center:]

            normalization = np.arange(num_bins, 0, -1).astype(np.float32)
            G2 = acf_right / (normalization + 1e-9)

            mean_I = np.mean(I_t)
            baseline = mean_I ** 2

            if baseline > 1e-9:
                g2_final = G2 / baseline
            else:
                g2_final = np.ones_like(G2)

            # 映射到 tau_grid
            indices = (self.tau_lags // self.dt_us).astype(np.int64)
            indices = np.clip(indices, 0, len(g2_final) - 1)
            g2_feature = g2_final[indices]

            g2_feature = np.nan_to_num(g2_feature, nan=1.0)
            log_intensity = np.log10(mean_I + 1e-6).astype(np.float32)

            # 直接存好 Tensor，训练时直接取
            self.processed_samples.append({
                'g2_curve': torch.from_numpy(g2_feature).float(),
                'aux_input': torch.tensor([log_intensity]).float(),
                'flow_label': torch.tensor([label]).float(),
                'k_factor': torch.tensor([m_val]).float()
            })

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        # 极速读取
        return self.processed_samples[idx]