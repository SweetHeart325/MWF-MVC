import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveletFilter(nn.Module):
    """自适应小波滤波器"""
    def __init__(self, K=10):
        super(WaveletFilter, self).__init__()
        self.K = K
        self.wavelet_coeffs = nn.Parameter(torch.randn(K) * 0.1)

    def haar_basis(self, eigenvals, scale):
        eigenvals = eigenvals.clone()
        eigenvals[eigenvals < 0] = 0
        start = 2.0 / self.K * scale
        end = start + 2.0 / self.K
        mask = torch.logical_and(eigenvals >= start, eigenvals < end)
        if end == 2.0:
            mask = torch.logical_and(eigenvals >= start, eigenvals <= end)
        return mask.float()

    def compute_response(self, eigenvals):
        theta = self.wavelet_coeffs
        theta = F.softplus(theta)
        theta = theta / (theta.max() + 1e-12)
        H = torch.zeros_like(eigenvals)
        for k in range(self.K):
            haar_mask = self.haar_basis(eigenvals, k)
            H += theta[k] * haar_mask
        return H

    def forward_with_eigen(self, eigenvals, eigenvecs):
        H = self.compute_response(eigenvals)
        filter_matrix = eigenvecs @ torch.diag(H) @ eigenvecs.T
        return filter_matrix

    def forward(self, L_sym):
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
        from utils import distance # 避免循环引用，局部导入
        embedding = filter_matrix.mm(xi.matmul(self.w1))
        embedding = torch.nn.functional.relu(embedding)
        embedding = filter_matrix.mm(embedding.matmul(self.w2))

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
        gae_params = []
        for gae in self.gae_list:
            gae_params.extend([gae.w1, gae.w2])
        return gae_params

    def get_wavelet_parameters(self):
        if not hasattr(self, 'wavelet_filters') or self.wavelet_filters is None:
            return []
        wavelet_params = []
        for wf in self.wavelet_filters:
            wavelet_params.extend(wf.parameters())
        return wavelet_params