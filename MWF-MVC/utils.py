import torch
import torch.nn as nn
import numpy as np


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
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


def get_symmetric_laplacian_from_weights(weights):
    device = weights.device
    I_n = torch.eye(weights.shape[0]).to(device)
    A_hat = get_Laplacian_from_weights(weights)
    return I_n - A_hat


def update_graph_with_eigen_cache(embedding_mv, num_neighbors):
    """更新图结构，同时缓存特征值分解结果"""
    with torch.no_grad():
        weights_mv, raw_weights_mv, target_filters = [], [], []
        eigenvals_list, eigenvecs_list = [], []

        for v in range(len(embedding_mv)):
            x = embedding_mv[v]
            weights, raw_weights = cal_weights_via_CAN(x.t(), num_neighbors)
            target_filter = get_Laplacian_from_weights(weights)
            lsym = get_symmetric_laplacian_from_weights(weights)

            # 特征分解
            eigenvals, eigenvecs = torch.linalg.eigh(lsym)

            weights_mv.append(weights)
            raw_weights_mv.append(raw_weights)
            target_filters.append(target_filter)
            eigenvals_list.append(eigenvals)
            eigenvecs_list.append(eigenvecs)

        return weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list


def compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters, wavelet_filters):
    total_loss = 0
    for i, (eigenvals, eigenvecs, target_filter, wavelet_filter) in enumerate(
            zip(eigenvals_list, eigenvecs_list, target_filters, wavelet_filters)):
        target_response = 1 - eigenvals
        wavelet_response = wavelet_filter.compute_response(eigenvals)
        spectral_loss = nn.MSELoss()(wavelet_response, target_response)
        total_loss += spectral_loss
    return total_loss / len(target_filters)