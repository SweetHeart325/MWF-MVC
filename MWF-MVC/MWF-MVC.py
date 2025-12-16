"""
Spectral Response Based Distillation with Wavelet Filter Substitution
实验流程：
1. 蒸馏阶段：预训练+微调，使用频率响应拟合损失，梯度只更新小波系数
2. 联合微调阶段：预训练+微调，用小波滤波器替换原滤波器，全参数更新
3. 两次评估对比性能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import scipy.io as sio
import warnings
import copy
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from sklearn import metrics, preprocessing
from scipy.optimize import linear_sum_assignment

import os
import pickle
from itertools import chain
import random

warnings.filterwarnings('ignore')
import scipy.sparse
import h5py


class WaveletFilter(nn.Module):
    """自适应小波滤波器"""

    def __init__(self, K=10):
        super(WaveletFilter, self).__init__()
        self.K = K
        self.wavelet_coeffs = nn.Parameter(torch.randn(K) * 0.1)

    def haar_basis(self, eigenvals, scale):
        """构造Haar小波基函数"""
        eigenvals = eigenvals.clone()
        eigenvals[eigenvals < 0] = 0

        start = 2.0 / self.K * scale
        end = start + 2.0 / self.K
        mask = torch.logical_and(eigenvals >= start, eigenvals < end)
        if end == 2.0:
            mask = torch.logical_and(eigenvals >= start, eigenvals <= end)

        return mask.float()

    def compute_response(self, eigenvals):
        """计算小波滤波器在给定特征值上的响应 H_θ(λ)"""
        theta = self.wavelet_coeffs
        theta = F.softplus(theta)
        theta = theta / (theta.max() + 1e-12)
        H = torch.zeros_like(eigenvals)
        for k in range(self.K):
            haar_mask = self.haar_basis(eigenvals, k)
            H += theta[k] * haar_mask
        return H

    def forward_with_eigen(self, eigenvals, eigenvecs):
        """使用预计算的特征值和特征向量构造滤波器矩阵"""
        H = self.compute_response(eigenvals)
        filter_matrix = eigenvecs @ torch.diag(H) @ eigenvecs.T
        return filter_matrix

    def forward(self, L_sym):
        """传统前向传播（如果需要独立计算）"""
        eigenvals, eigenvecs = torch.linalg.eigh(L_sym)
        return self.forward_with_eigen(eigenvals, eigenvecs)


class AdaGAE(torch.nn.Module):
    """图自编码器"""

    def __init__(self, layer_dims, use_wavelet=False, wavelet_filter=None):
        super().__init__()
        self.w1 = self.get_weight_initial([layer_dims[0], layer_dims[1]])
        self.w2 = self.get_weight_initial([layer_dims[1], layer_dims[2]])
        self.use_wavelet = use_wavelet
        self.wavelet_filter = wavelet_filter

    def get_weight_initial(self, shape):
        bound = np.sqrt(6.0 / (shape[0] + shape[1]))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def forward(self, xi, filter_matrix):
        """filter_matrix: 预计算好的滤波器矩阵"""
        embedding = filter_matrix.mm(xi.matmul(self.w1))
        embedding = torch.nn.functional.relu(embedding)
        embedding = filter_matrix.mm(embedding.matmul(self.w2))

        # 重构
        distances = distance(embedding.t(), embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return embedding, recons_w + 1e-10

    def cal_loss(self, raw_weights, recons, weights, embeding, lam):
        re_loss = raw_weights * torch.log(raw_weights / recons + 1e-10)
        re_loss = re_loss.sum(dim=1).mean()

        size = embeding.shape[0]
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        tr_loss = torch.trace(embeding.t().matmul(L).matmul(embeding)) / size
        return re_loss, tr_loss


class AdaGAEMV(torch.nn.Module):
    """多视图图自编码器"""

    def __init__(self, X, layers, device, use_wavelet=False, wavelet_K=10):
        super().__init__()
        layers_list = [[x.shape[1]] + layers for x in X]
        self.use_wavelet = use_wavelet

        if use_wavelet:
            self.wavelet_filters = nn.ModuleList([
                WaveletFilter(K=wavelet_K).to(device) for _ in X
            ])

        self.gae_list = [
            AdaGAE(layer, use_wavelet=use_wavelet,
                   wavelet_filter=self.wavelet_filters[i] if use_wavelet else None).to(device)
            for i, layer in enumerate(layers_list)
        ]

    def forward(self, X, filter_matrix_list):
        """filter_matrix_list: 预计算好的滤波器矩阵列表"""
        embedding_list = []
        recons_w_list = []
        for i in range(len(X)):
            embedding, recons_w = self.gae_list[i](X[i], filter_matrix_list[i])
            embedding_list.append(embedding)
            recons_w_list.append(recons_w)
        return embedding_list, recons_w_list

    def cal_loss(self, raw_weights_mv, recons_w_list, weights_mv, embedding_list, lam):
        re_loss_list = []
        tr_loss_list = []
        for i in range(len(recons_w_list)):
            re_loss, tr_loss = self.gae_list[i].cal_loss(
                raw_weights_mv[i],
                recons_w_list[i],
                weights_mv[i],
                embedding_list[i],
                lam,
            )
            re_loss_list.append(re_loss)
            tr_loss_list.append(tr_loss)
        re_loss = sum(re_loss_list) / len(re_loss_list)
        tr_loss = sum(tr_loss_list) / len(tr_loss_list)
        return re_loss, tr_loss

    def get_gae_parameters(self):
        """获取GAE参数（w1, w2）"""
        gae_params = []
        for gae in self.gae_list:
            gae_params.extend([gae.w1, gae.w2])
        return gae_params

    def get_wavelet_parameters(self):
        """获取小波滤波器参数"""
        if not hasattr(self, 'wavelet_filters') or self.wavelet_filters is None:
            return []
        wavelet_params = []
        for wf in self.wavelet_filters:
            wavelet_params.extend(wf.parameters())
        return wavelet_params


def compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters, wavelet_filters):
    """
    计算频率响应拟合损失
    eigenvals_list: 每个视图的Lsym特征值
    eigenvecs_list: 每个视图的Lsym特征向量
    target_filters: 目标滤波器列表 (A^)
    wavelet_filters: 小波滤波器列表
    """
    total_loss = 0
    for i, (eigenvals, eigenvecs, target_filter, wavelet_filter) in enumerate(
            zip(eigenvals_list, eigenvecs_list, target_filters, wavelet_filters)):
        # 目标滤波器的理论频率响应: H_T(λ) = 1 - λ
        # 因为 A^ = I - Lsym，所以 A^ 在特征值λ上的响应就是 1-λ
        target_response = 1 - eigenvals

        # 小波滤波器的响应: H_θ(λ)
        wavelet_response = wavelet_filter.compute_response(eigenvals)

        # 计算频率响应差异
        spectral_loss = nn.MSELoss()(wavelet_response, target_response)
        total_loss += spectral_loss

    return total_loss / len(target_filters)


def distance(X, Y, square=True):
    """计算欧式距离"""
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    """计算CAN权重"""
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 1e-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))

    T = top_k - distances
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    weights = weights.relu().cpu()

    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])

    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    """计算归一化邻接矩阵 A^"""
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


def get_symmetric_laplacian_from_weights(weights):
    """计算对称拉普拉斯矩阵 Lsym = I - A^"""
    device = weights.device
    I_n = torch.eye(weights.shape[0]).to(device)
    A_hat = get_Laplacian_from_weights(weights)  # A^
    return I_n - A_hat  # Lsym = I - A^


def update_graph_with_eigen_cache(embedding_mv, num_neighbors):
    """
    更新图结构，同时缓存特征值分解结果
    返回: weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list
    """
    with torch.no_grad():
        weights_mv, raw_weights_mv, target_filters = [], [], []
        eigenvals_list, eigenvecs_list = [], []

        for v in range(len(embedding_mv)):
            x = embedding_mv[v]
            weights, raw_weights = cal_weights_via_CAN(x.t(), num_neighbors)

            # 目标滤波器 (A^)
            target_filter = get_Laplacian_from_weights(weights)

            # 对称拉普拉斯矩阵 (Lsym = I - A^)
            lsym = get_symmetric_laplacian_from_weights(weights)

            # 特征值分解 Lsym（一次分解，缓存结果）
            eigenvals, eigenvecs = torch.linalg.eigh(lsym)

            weights_mv.append(weights)
            raw_weights_mv.append(raw_weights)
            target_filters.append(target_filter)
            eigenvals_list.append(eigenvals)
            eigenvecs_list.append(eigenvecs)

        return weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list


def update_graph(embedding_mv, num_neighbors):
    """原始的图更新函数（不缓存特征分解）"""
    with torch.no_grad():
        weights_mv, raw_weights_mv, target_filters = [], [], []
        for v in range(len(embedding_mv)):
            x = embedding_mv[v]
            weights, raw_weights = cal_weights_via_CAN(x.t(), num_neighbors)
            target_filter = get_Laplacian_from_weights(weights)

            weights_mv.append(weights)
            raw_weights_mv.append(raw_weights)
            target_filters.append(target_filter)

        return weights_mv, raw_weights_mv, target_filters


def load_new_format_data(dataset_name):
    file_dir = os.path.dirname(__file__)
    data_path = f"{file_dir}/dataset/{dataset_name}.mat"

    try:
        data = sio.loadmat(data_path)
        is_h5 = False
    except NotImplementedError:
        data = h5py.File(data_path, "r")
        is_h5 = True

    keymap = {k.lower(): k for k in data.keys()}

    def resolve(obj, h5_file):
        if isinstance(obj, h5py.Dataset):
            obj = obj[()]
        if isinstance(obj, (h5py.Reference, getattr(h5py.h5r, "Reference", object))):
            return resolve(h5_file[obj], h5_file)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            return [resolve(item, h5_file) for item in obj.flat]
        if isinstance(obj, (list, tuple)):
            return [resolve(item, h5_file) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.T if obj.ndim > 1 else obj
        return obj

    # 特殊处理：Fashion 和 MNIST-USPS 数据集
    if dataset_name.lower() == "fashion":
        if is_h5:
            Y = resolve(data['Y'], data).astype(np.int32).reshape(10000, )
            data1 = resolve(data['X1'], data).astype(np.float32)
            data2 = resolve(data['X2'], data).astype(np.float32)
            data3 = resolve(data['X3'], data).astype(np.float32)
        else:
            Y = data['Y'].astype(np.int32).reshape(10000, )
            data1 = data['X1'].astype(np.float32)
            data2 = data['X2'].astype(np.float32)
            data3 = data['X3'].astype(np.float32)
        X = [data1, data2, data3]
        return X, Y

    elif dataset_name.lower() in ["mnist_usps", "mnist-usps"]:
        if is_h5:
            Y = resolve(data['Y'], data).astype(np.int32).reshape(5000, )
            data1 = resolve(data['X1'], data).astype(np.float32).reshape(5000, -1)
            data2 = resolve(data['X2'], data).astype(np.float32).reshape(5000, -1)
        else:
            Y = data['Y'].astype(np.int32).reshape(5000, )
            data1 = data['X1'].astype(np.float32).reshape(5000, -1)
            data2 = data['X2'].astype(np.float32).reshape(5000, -1)
        X = [data1, data2]
        return X, Y

    # 通用处理：其他数据集
    if "X" in data:
        X = data["X"]
    elif "x" in keymap:
        X = data[keymap["x"]]
    elif "xs" in keymap:
        X = data[keymap["xs"]]
    else:
        raise KeyError(f"'{data_path}' 里找不到数据矩阵 X/x/Xs，可用键有：{list(data.keys())}")

    if "y" in keymap:
        Y = data[keymap["y"]]
    elif "Y" in data:
        Y = data["Y"]
    else:
        raise KeyError(f"'{data_path}' 里找不到标签 y/Y，可用键有：{list(data.keys())}")

    if is_h5:
        X = resolve(X, data)
        Y = resolve(Y, data)
    else:
        if isinstance(X, np.ndarray) and X.dtype == object:
            X = X[0] if X.shape[0] == 1 else X[:, 0]

    if not isinstance(X, list):
        if isinstance(X, np.ndarray) and X.dtype == object:
            X = list(X)
        else:
            X = [X]

    return X, Y


def normalize_feature(X, dataset_name):
    """对多视图数据进行归一化处理，支持稀疏矩阵和密集矩阵，并处理维度问题"""
    normalized_X = []
    for i, x in enumerate(X):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # 确保数据是2D的
        if x.ndim != 2:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 4:
                x = np.squeeze(x, axis=-1)
                x = x.reshape(x.shape[0], -1)
            elif x.ndim == 3:
                x = x.reshape(x.shape[0], -1)
            else:
                x = x.reshape(x.shape[0], -1)

        # 检查是否有无效值
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # 检查是否为稀疏矩阵
        if scipy.sparse.issparse(x):
            x_normalized = preprocessing.scale(x, with_mean=False)
            normalized_X.append(x_normalized)
        else:
            try:
                if dataset_name in ["coil20mv", "hdigit", "handwritten_2v"]:
                    scaler = preprocessing.StandardScaler()
                    x_normalized = scaler.fit_transform(x)
                else:
                    x_normalized = preprocessing.scale(x)
                normalized_X.append(x_normalized)
            except Exception as e:
                try:
                    scaler = preprocessing.MinMaxScaler()
                    x_normalized = scaler.fit_transform(x)
                    normalized_X.append(x_normalized)
                except Exception as e2:
                    normalized_X.append(x.astype(np.float32))

    return normalized_X


def load_dataset(dataset_name):
    X, Y = load_new_format_data(dataset_name)
    Y = np.array(Y.astype(int))
    if Y.ndim > 1:
        Y = Y.flatten()
    if Y.min() == 1:
        Y = Y - 1

    n_cluster = len(np.unique(Y))
    n_sample = len(Y)
    n_view = len(X)

    X = normalize_feature(X, dataset_name)
    return X, Y, n_cluster, n_sample, n_view


def compute_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size * 100


def compute_fscore(labels_true, labels_pred):
    from sklearn.metrics.cluster._supervised import check_clusterings

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    if labels_true.shape == (0,):
        raise ValueError("input labels must not be empty.")

    n_samples = len(labels_true)
    true_clusters = {}
    pred_clusters = {}

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0
    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    if precision + recall == 0:
        return 0.0

    f_score = 2 * precision * recall / (precision + recall)
    return f_score


def cluster_one_time(features, labels, n_clusters):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10)
    pred = km.fit_predict(features)

    labels = np.reshape(labels, np.shape(pred))
    if np.min(labels) == 1:
        labels -= 1

    nmi = nmi_score(labels, pred) * 100
    acc = compute_acc(labels, pred)
    fscore = compute_fscore(labels, pred) * 100

    return round(nmi, 2), round(acc, 2), round(fscore, 2)


def cluster_eval(n_clusters, features, labels, count=1, desc="cluster"):
    nmi_array, acc_array, f1_array = [], [], []

    for _ in range(count):
        nmi, acc, fscore = cluster_one_time(features, labels, n_clusters)
        nmi_array.append(nmi)
        acc_array.append(acc)
        f1_array.append(fscore)

    nmi_avg, nmi_std = round(np.mean(nmi_array), 2), round(np.std(nmi_array), 2)
    acc_avg, acc_std = round(np.mean(acc_array), 2), round(np.std(acc_array), 2)
    f1_avg, f1_std = round(np.mean(f1_array), 2), round(np.std(f1_array), 2)

    results = [nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std]
    return results


def cluster_by_multi_ways(z_list, filter_list, Y, n_cluster, count=1, fusion_kind="concat_avg",
                          view_index=0):
    """多视图聚类评估"""
    if fusion_kind == "concat":
        z = torch.hstack(z_list).detach().cpu().numpy()
        return cluster_eval(n_cluster, z, Y, count=count, desc=fusion_kind)

    elif fusion_kind == "concat_avg":
        # 单视图滤波
        new_z = torch.matmul(filter_list[view_index], z_list[view_index])
        new_z = new_z.detach().cpu().numpy()
        cluster_eval(n_cluster, new_z, Y, desc=f"X{view_index}")

        # 多视图融合滤波
        z = torch.hstack(z_list)
        L = torch.mean(torch.stack(filter_list), dim=0)
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        results = cluster_eval(n_cluster, new_z, Y, count=count, desc=fusion_kind)
        return results


def evaluate_model(model, embedding_list, target_filters, eigenvals_list, eigenvecs_list, Y, n_cluster, args,
                   fusion_kind="concat_avg"):
    """评估模型性能"""
    with torch.no_grad():
        model.eval()

        neighbor_num = args['neighbor_max']

        if model.use_wavelet:
            # 小波模型：需要特征分解来构造滤波器
            wavelet_filter_matrices = []
            for i, (eigenvals, eigenvecs, wavelet_filter) in enumerate(
                    zip(eigenvals_list, eigenvecs_list, model.wavelet_filters)):
                filter_matrix = wavelet_filter.forward_with_eigen(eigenvals, eigenvecs)
                wavelet_filter_matrices.append(filter_matrix)

            filter_for_eval = wavelet_filter_matrices
        else:
            filter_for_eval = target_filters

        results = cluster_by_multi_ways(
            embedding_list, filter_for_eval, Y, n_cluster, count=10, fusion_kind=fusion_kind
        )

        nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std = results
        return nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std


def train_distillation_stage(X_tensor_list, Y, n_cluster, n_sample, n_view, args, device):
    """蒸馏阶段：让小波滤波器学习目标滤波器的频率响应"""
    print("=== Stage 1: Distillation Stage ===")

    # 创建基础模型（用于产生目标滤波器）
    base_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=False).to(device)

    # 创建小波模型（用于学习拟合）
    wavelet_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=True,
                             wavelet_K=args.get('wavelet_K', 10)).to(device)

    # 创建分离的优化器
    main_optimizer = torch.optim.Adam(base_model.get_gae_parameters(), lr=args["learning_rate"])
    wavelet_optimizer = torch.optim.Adam(wavelet_model.get_wavelet_parameters(), lr=args["learning_rate"])

    # 初始化图结构
    neighbor_num = args["neighbor_init"]
    weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
        X_tensor_list, neighbor_num)

    # 蒸馏预训练阶段
    for epoch in range(args["pretrain_epoch"]):
        for i in range(args["pretrain_iter"]):
            # 主损失计算和更新
            embedding_list, recons_w_list = base_model(X_tensor_list, target_filters)
            re_loss, tr_loss = base_model.cal_loss(
                raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
            )
            main_loss = re_loss + args["lam_tr"] * tr_loss

            main_optimizer.zero_grad()
            main_loss.backward()
            main_optimizer.step()

            # 拟合损失计算和更新
            spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                          wavelet_model.wavelet_filters)
            fitting_loss = args.get("beta", 0.1) * spectral_loss

            wavelet_optimizer.zero_grad()
            fitting_loss.backward()
            wavelet_optimizer.step()

            if (i + 1) % args["log_freq"] == 0:
                print(f'Epoch[{epoch + 1}/{args["pretrain_epoch"]}], Step [{i + 1}/{args["pretrain_iter"]}], '
                      f'Main_L: {main_loss.item():.4f}, Spec_L: {spectral_loss.item():.6f}')

        weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
            embedding_list, neighbor_num)

        # 自步增邻
        if neighbor_num < args["neighbor_max"]:
            neighbor_num = min(neighbor_num + args["neighbor_incr"], args["neighbor_max"])

    # 蒸馏微调阶段
    mse_loss_func = nn.MSELoss()
    for epoch in range(args["finetune_epoch"]):
        # 主损失计算和更新
        embedding_list, recons_w_list = base_model(X_tensor_list, target_filters)
        re_loss, tr_loss = base_model.cal_loss(
            raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
        )
        con_loss = 0
        for vi in range(n_view):
            for vj in range(vi + 1, n_view):
                con_loss += mse_loss_func(embedding_list[vi], embedding_list[vj])
        main_loss = re_loss + args["lam_tr"] * tr_loss + args["lam_con"] * con_loss

        main_optimizer.zero_grad()
        main_loss.backward()
        main_optimizer.step()

        # 拟合损失计算和更新
        spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                      wavelet_model.wavelet_filters)
        fitting_loss = args.get("beta", 0.1) * spectral_loss

        wavelet_optimizer.zero_grad()
        fitting_loss.backward()
        wavelet_optimizer.step()

        if (epoch + 1) % args["log_freq"] == 0:
            print(f'Epoch[{epoch + 1}/{args["finetune_epoch"]}], '
                  f'Main_L: {main_loss.item():.4f}, Spec_L: {spectral_loss.item():.6f}')

    # 蒸馏阶段评估
    print("\n=== Evaluating After Distillation Stage ===")
    nmi, nmi_std, acc, acc_std, f1, f1_std = evaluate_model(base_model, embedding_list, target_filters, eigenvals_list,
                                                            eigenvecs_list, Y, n_cluster, args)
    distillation_results = (acc, acc_std, nmi, nmi_std, f1, f1_std)
    print(f"Distillation Stage - ACC: {acc:.2f}±{acc_std:.2f}, NMI: {nmi:.2f}±{nmi_std:.2f}, F1: {f1:.2f}±{f1_std:.2f}")

    return wavelet_model, distillation_results


def train_joint_finetuning_stage(X_tensor_list, Y, n_cluster, n_sample, n_view,
                                 trained_wavelet_model, args, device):
    """联合微调阶段：用小波滤波器替换目标滤波器，全参数更新"""
    print("\n=== Stage 2: Joint Fine-tuning Stage ===")

    # 创建使用小波滤波器的模型
    joint_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=True,
                           wavelet_K=args.get('wavelet_K', 10)).to(device)

    # 复制蒸馏阶段学习好的小波滤波器参数
    for i, (target_wf, source_wf) in enumerate(
            zip(joint_model.wavelet_filters, trained_wavelet_model.wavelet_filters)):
        target_wf.wavelet_coeffs.data = source_wf.wavelet_coeffs.data.clone()

    # 全参数优化器
    all_params = joint_model.get_gae_parameters() + joint_model.get_wavelet_parameters()
    joint_optimizer = torch.optim.Adam(all_params, lr=args["learning_rate"])

    # 重新开始完整的训练流程
    neighbor_num = args["neighbor_init"]
    weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
        X_tensor_list, neighbor_num)

    # 联合预训练阶段
    for epoch in range(args["pretrain_epoch"]):
        for i in range(args["pretrain_iter"]):
            wavelet_filter_matrices = []
            # 使用小波滤波器矩阵进行前向传播
            for j, (eigenvals, eigenvecs, wavelet_filter) in enumerate(
                    zip(eigenvals_list, eigenvecs_list, joint_model.wavelet_filters)):
                filter_matrix = wavelet_filter.forward_with_eigen(eigenvals, eigenvecs)
                wavelet_filter_matrices.append(filter_matrix)

            embedding_list, recons_w_list = joint_model(X_tensor_list, wavelet_filter_matrices)
            re_loss, tr_loss = joint_model.cal_loss(
                raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
            )

            # 频率响应拟合损失
            spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                          joint_model.wavelet_filters)

            # 总损失
            total_loss = re_loss + args["lam_tr"] * tr_loss + args.get("beta", 0.1) * spectral_loss

            joint_optimizer.zero_grad()
            total_loss.backward()
            joint_optimizer.step()

            if (i + 1) % args["log_freq"] == 0:
                print(f'Epoch[{epoch + 1}/{args["pretrain_epoch"]}], Step [{i + 1}/{args["pretrain_iter"]}], '
                      f'Total_L: {total_loss.item():.4f}')

        weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
            embedding_list, neighbor_num)

        # 自步增邻
        if neighbor_num < args["neighbor_max"]:
            neighbor_num = min(neighbor_num + args["neighbor_incr"], args["neighbor_max"])

    # 联合微调阶段
    mse_loss_func = nn.MSELoss()
    for epoch in range(args["finetune_epoch"]):
        wavelet_filter_matrices = []
        for j, (eigenvals, eigenvecs, wavelet_filter) in enumerate(
                zip(eigenvals_list, eigenvecs_list, joint_model.wavelet_filters)):
            filter_matrix = wavelet_filter.forward_with_eigen(eigenvals, eigenvecs)
            wavelet_filter_matrices.append(filter_matrix)

        embedding_list, recons_w_list = joint_model(X_tensor_list, wavelet_filter_matrices)
        re_loss, tr_loss = joint_model.cal_loss(
            raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
        )

        # 一致性损失
        con_loss = 0
        for vi in range(n_view):
            for vj in range(vi + 1, n_view):
                con_loss += mse_loss_func(embedding_list[vi], embedding_list[vj])

        # 频率响应拟合损失
        spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                      joint_model.wavelet_filters)

        # 总损失
        total_loss = re_loss + args["lam_tr"] * tr_loss + args["lam_con"] * con_loss + args.get("beta",
                                                                                                0.1) * spectral_loss

        joint_optimizer.zero_grad()
        total_loss.backward()
        joint_optimizer.step()

        if (epoch + 1) % args["log_freq"] == 0:
            print(f'Epoch[{epoch + 1}/{args["finetune_epoch"]}], '
                  f'Total_L: {total_loss.item():.4f}')

    print("\n=== Evaluating After Joint Stage ===")
    nmi, nmi_std, acc, acc_std, f1, f1_std = evaluate_model(joint_model, embedding_list, target_filters, eigenvals_list,
                                                            eigenvecs_list, Y, n_cluster, args)
    joint_results = (acc, acc_std, nmi, nmi_std, f1, f1_std)
    print(f"Joint Stage - ACC: {acc:.2f}±{acc_std:.2f}, NMI: {nmi:.2f}±{nmi_std:.2f}, F1: {f1:.2f}±{f1_std:.2f}")

    return joint_model, joint_results


def spectral_distillation_experiment(dataset_name, args):
    """完整的频谱蒸馏实验"""
    print(f"\n{'=' * 80}")
    print(f"SPECTRAL DISTILLATION WAVELET SUBSTITUTION EXPERIMENT")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    X_list, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)

    X_tensor_list = []
    for i, X in enumerate(X_list):
        if scipy.sparse.issparse(X):
            X_dense = X.toarray()
            X_tensor = torch.from_numpy(X_dense).float().to(device)
        elif isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(device)
        else:
            X_tensor = torch.from_numpy(X).float().to(device)
        X_tensor_list.append(X_tensor)

    print(f"Samples: {n_sample}, Views: {n_view}, Classes: {n_cluster}")

    # 阶段1：蒸馏阶段
    trained_wavelet_model, distillation_results = train_distillation_stage(
        X_tensor_list, Y, n_cluster, n_sample, n_view, args, device
    )

    # 阶段2：联合微调阶段
    joint_model, joint_results = train_joint_finetuning_stage(
        X_tensor_list, Y, n_cluster, n_sample, n_view, trained_wavelet_model, args, device
    )

    # 性能对比分析
    print(f"\n{'=' * 80}")
    print("FINAL PERFORMANCE COMPARISON")
    print(f"{'=' * 80}")

    # 解包结果
    dist_acc, dist_acc_std, dist_nmi, dist_nmi_std, dist_f1, dist_f1_std = distillation_results
    joint_acc, joint_acc_std, joint_nmi, joint_nmi_std, joint_f1, joint_f1_std = joint_results

    acc_diff = joint_acc - dist_acc
    nmi_diff = joint_nmi - dist_nmi
    f1_diff = joint_f1 - dist_f1

    print(f"{'Method':<35} {'ACC':<15} {'NMI':<15} {'F1':<15}")
    print("-" * 85)
    print(
        f"{'Distillation Stage':<35} {dist_acc:.2f}±{dist_acc_std:.2f}{'':<6} {dist_nmi:.2f}±{dist_nmi_std:.2f}{'':<6} {dist_f1:.2f}±{dist_f1_std:.2f}")
    print(
        f"{'Joint Fine-tuning Stage':<35} {joint_acc:.2f}±{joint_acc_std:.2f}{'':<6} {joint_nmi:.2f}±{joint_nmi_std:.2f}{'':<6} {joint_f1:.2f}±{joint_f1_std:.2f}")
    print(f"{'Performance Difference':<35} {acc_diff:+.2f}{'':<10} {nmi_diff:+.2f}{'':<10} {f1_diff:+.2f}")

    # 框架优势评估
    if acc_diff > 1.0:
        framework_quality = "SUPERIOR"
        advantage_desc = f"{acc_diff:+.2f}% improvement - Spectral distillation works excellently!"
    elif acc_diff > 0:
        framework_quality = "IMPROVED"
        advantage_desc = f"{acc_diff:+.2f}% slight improvement"
    elif abs(acc_diff) <= 1.0:
        framework_quality = "COMPARABLE"
        advantage_desc = f"{acc_diff:+.2f}% minimal difference - good spectral fitting"
    else:
        framework_quality = "INFERIOR"
        advantage_desc = f"{acc_diff:+.2f}% performance loss"

    print(f"\nSpectral Distillation Framework: {framework_quality}")
    print(f"Performance Summary: {advantage_desc}")

    return {
        'distillation': {
            'acc': dist_acc, 'acc_std': dist_acc_std,
            'nmi': dist_nmi, 'nmi_std': dist_nmi_std,
            'f1': dist_f1, 'f1_std': dist_f1_std,
        },
        'joint_finetuning': {
            'acc': joint_acc, 'acc_std': joint_acc_std,
            'nmi': joint_nmi, 'nmi_std': joint_nmi_std,
            'f1': joint_f1, 'f1_std': joint_f1_std,
        },
        'diff': {
            'acc': acc_diff, 'nmi': nmi_diff, 'f1': f1_diff
        },
        'framework_quality': framework_quality,
        'advantage_desc': advantage_desc
    }


def main():
    def set_seed(seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(1234)
    print("=== Spectral Distillation Wavelet Substitution Experiment ===")

    # 数据集配置
    dataset_configs = {
        '100leaves': {
            'neighbor_max': 16,
            'lam_tr': 0.1,
            'lam_con': 0.1,
            'beta': 0.1,
        },
        'coil20mv': {
            'neighbor_max': 72,
            'lam_tr': 1,
            'lam_con': 1,
            'beta': 0.1,
        },
        'handwritten_2v': {
            'neighbor_max': 200,
            'lam_tr': 0.001,
            'lam_con': 0.001,
            'beta': 0.1,
        },
        'msrcv1_6v': {
            'neighbor_max': 30,
            'lam_tr': 0.01,
            'lam_con': 0.001,
            'beta': 0.7,
        },
        "ORL": {
            "neighbor_max": 10,
            "lam_tr": 1,
            "lam_con": 0.001,
            'beta': 0.7,
        },
        'yale': {
            "neighbor_max": 11,
            "lam_tr": 1,
            "lam_con": 10,
            'beta': 0.5,
        },
        "Fashion": {
            "neighbor_max": 1000,
            "lam_tr": 0.1,
            "lam_con": 10,
            'beta': 0.5,
        },
    }

    # 固定训练参数
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

    # 测试数据集
    test_datasets = [
        "100leaves",
        "coil20mv",
        "handwritten_2v",
        "msrcv1_6v",
        "ORL",
        'yale',
        'Fashion',
    ]

    results = {}
    for dataset_name in test_datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            args = {**fixed_args, **dataset_configs[dataset_name]}
            result = spectral_distillation_experiment(dataset_name, args)
            results[dataset_name] = result

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 实验总结
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    if results:
        for dataset, result in results.items():
            print(f"\nDataset: {dataset}")
            print(f"{'=' * 60}")

            distillation = result['distillation']
            joint_ft = result['joint_finetuning']

            print(
                f"Distillation Stage:      ACC={distillation['acc']:.2f}, NMI={distillation['nmi']:.2f}, F1={distillation['f1']:.2f}")
            print(
                f"Joint Fine-tuning:       ACC={joint_ft['acc']:.2f}, NMI={joint_ft['nmi']:.2f}, F1={joint_ft['f1']:.2f}")

            # 阶段间改进分析
            stage_acc_diff = joint_ft['acc'] - distillation['acc']
            stage_nmi_diff = joint_ft['nmi'] - distillation['nmi']
            stage_f1_diff = joint_ft['f1'] - distillation['f1']

            print(
                f"Joint vs Distillation:   ACC={stage_acc_diff:+.2f}, NMI={stage_nmi_diff:+.2f}, F1={stage_f1_diff:+.2f}")
            print(f"Framework Quality:       {result['framework_quality']}")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()