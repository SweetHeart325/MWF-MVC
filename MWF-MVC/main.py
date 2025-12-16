import random
import numpy as np
import torch
import warnings
from trainer import spectral_fitting_experiment

warnings.filterwarnings('ignore')


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(1234)
    print("=== Spectral Fitting Wavelet Substitution Experiment ===")

    dataset_configs = {
        '100leaves': {'neighbor_max': 16, 'lam_tr': 0.1, 'lam_con': 0.1, 'beta': 0.1},
        'coil20mv': {'neighbor_max': 72, 'lam_tr': 1, 'lam_con': 1, 'beta': 0.1},
        # 可以在这里添加更多数据集配置...
    }

    fixed_args = {
        'log_freq': 20,
        'learning_rate': 1e-3,
        'layers': [256, 64],
        'pretrain_iter': 100,
        'pretrain_epoch': 10,
        'finetune_epoch': 100,
        'wavelet_K': 10,
        'neighbor_init': 6,
        'neighbor_incr': 5,
    }

    test_datasets = ["100leaves"]  # 修改这里以测试不同数据集

    for dataset_name in test_datasets:
        if dataset_name not in dataset_configs:
            print(f"Skipping {dataset_name}: No config found.")
            continue

        try:
            args = {**fixed_args, **dataset_configs[dataset_name]}
            spectral_fitting_experiment(dataset_name, args)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()