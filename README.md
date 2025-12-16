# Spectral Fitting with Wavelet Filter Substitution for Multi-View Clustering

本项目实现了一种基于频谱拟合（Spectral Fitting）和小波滤波器替换（Wavelet Filter Substitution）的多视图聚类算法。

该方法主要包含两个阶段：
1.  **频谱拟合阶段 (Spectral Fitting Stage)**：让自适应小波滤波器学习目标图滤波器（如对称拉普拉斯矩阵）的频率响应。
2.  **联合微调阶段 (Joint Fine-tuning Stage)**：用学习好的小波滤波器替换原滤波器，并进行全参数联合优化。

## 📂 项目结构

建议的项目文件结构如下：

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
└── README.md               # 项目说明文档

##环境依赖
pip install -r requirements.txt
主要依赖库包括：

PyTorch (用于深度学习模型构建)

NumPy & SciPy (用于矩阵运算和稀疏矩阵处理)

scikit-learn (用于聚类评估指标计算)

h5py (用于加载部分 .mat 数据格式)

##运行方式
##直接运行 main.py 即可开始实验：


python main.py
