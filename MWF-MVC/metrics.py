import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment

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
    return [nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std]

def cluster_by_multi_ways(z_list, filter_list, Y, n_cluster, count=1, fusion_kind="concat_avg", view_index=0):
    if fusion_kind == "concat":
        z = torch.hstack(z_list).detach().cpu().numpy()
        return cluster_eval(n_cluster, z, Y, count=count, desc=fusion_kind)
    elif fusion_kind == "concat_avg":
        # 单视图滤波部分省略，只保留融合滤波
        z = torch.hstack(z_list)
        L = torch.mean(torch.stack(filter_list), dim=0)
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        return cluster_eval(n_cluster, new_z, Y, count=count, desc=fusion_kind)
    return None