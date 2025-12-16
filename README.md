# Multi-Scale Spatial-Spectral Filtering for Robust Multi-View Clustering

本项目实现了一种基于频谱拟合（Spectral Fitting）和小波滤波器替换（Wavelet Filter Substitution）的多视图聚类算法。

该方法主要包含两个阶段：
1.  **频谱拟合阶段 (Spectral Fitting Stage)**：让自适应小波滤波器学习目标图滤波器（如对称拉普拉斯矩阵）的频率响应。
2.  **联合微调阶段 (Joint Fine-tuning Stage)**：用学习好的小波滤波器替换原滤波器，并进行全参数联合优化。

## 📂 项目结构

项目文件结构如下：

```text
project_root/
│
├── dataset/                # 数据集文件夹
│   ├── 100leaves.mat       # 示例数据文件
│   └── ...
│
├── dataloader.py           # 数据加载与预处理
├── models.py               # 模型定义 (WaveletFilter, AdaGAE, AdaGAEMV)
├── utils.py                # 通用工具 (图构建、距离计算、特征分解)
├── metrics.py              # 评估指标 (ACC, NMI, F1)
├── trainer.py              # 训练流程逻辑 (频谱拟合与联合微调)
├── main.py                 # 程序入口与配置
│
├── requirements.txt        # 依赖库列表
└── README.md               # 项目说明文
```

## 🛠️ 环境依赖

请运行以下命令安装所需的依赖库：

```bash
pip install -r requirements.txt
```

## ▶️ 运行方式

```bash
python main.py
```
