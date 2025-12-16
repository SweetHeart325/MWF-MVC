import torch
import torch.nn as nn
import scipy.sparse
import numpy as np

from models import AdaGAEMV
from utils import update_graph_with_eigen_cache, compute_spectral_fitting_loss
from metrics import cluster_by_multi_ways
from dataloader import load_dataset

def evaluate_model(model, embedding_list, target_filters, eigenvals_list, eigenvecs_list, Y, n_cluster, args, fusion_kind="concat_avg"):
    with torch.no_grad():
        model.eval()
        if model.use_wavelet:
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
        return results # nmi, std, acc, std, f1, std

def train_spectral_fitting_stage(X_tensor_list, Y, n_cluster, n_sample, n_view, args, device):
    print("=== Stage 1: Spectral Fitting Stage ===")
    base_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=False).to(device)
    wavelet_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=True,
                             wavelet_K=args.get('wavelet_K', 10)).to(device)

    main_optimizer = torch.optim.Adam(base_model.get_gae_parameters(), lr=args["learning_rate"])
    wavelet_optimizer = torch.optim.Adam(wavelet_model.get_wavelet_parameters(), lr=args["learning_rate"])

    neighbor_num = args["neighbor_init"]
    weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
        X_tensor_list, neighbor_num)

    # 预训练
    for epoch in range(args["pretrain_epoch"]):
        for i in range(args["pretrain_iter"]):
            embedding_list, recons_w_list = base_model(X_tensor_list, target_filters)
            re_loss, tr_loss = base_model.cal_loss(
                raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
            )
            main_loss = re_loss + args["lam_tr"] * tr_loss
            main_optimizer.zero_grad()
            main_loss.backward()
            main_optimizer.step()

            spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                          wavelet_model.wavelet_filters)
            fitting_loss = args.get("beta", 0.1) * spectral_loss
            wavelet_optimizer.zero_grad()
            fitting_loss.backward()
            wavelet_optimizer.step()

        weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
            embedding_list, neighbor_num)
        if neighbor_num < args["neighbor_max"]:
            neighbor_num = min(neighbor_num + args["neighbor_incr"], args["neighbor_max"])

    # 微调
    mse_loss_func = nn.MSELoss()
    for epoch in range(args["finetune_epoch"]):
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

        spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                      wavelet_model.wavelet_filters)
        fitting_loss = args.get("beta", 0.1) * spectral_loss
        wavelet_optimizer.zero_grad()
        fitting_loss.backward()
        wavelet_optimizer.step()

        if (epoch + 1) % args["log_freq"] == 0:
            print(f'Epoch[{epoch + 1}/{args["finetune_epoch"]}], Main_L: {main_loss.item():.4f}, Spec_L: {spectral_loss.item():.6f}')

    print("\n=== Evaluating After Spectral Fitting Stage ===")
    nmi, nmi_std, acc, acc_std, f1, f1_std = evaluate_model(base_model, embedding_list, target_filters, eigenvals_list,
                                                            eigenvecs_list, Y, n_cluster, args)
    fitting_results = (acc, acc_std, nmi, nmi_std, f1, f1_std)
    print(f"Spectral Fitting Stage - ACC: {acc:.2f}, NMI: {nmi:.2f}, F1: {f1:.2f}")
    return wavelet_model, fitting_results

def train_joint_finetuning_stage(X_tensor_list, Y, n_cluster, n_sample, n_view, trained_wavelet_model, args, device):
    print("\n=== Stage 2: Joint Fine-tuning Stage ===")
    joint_model = AdaGAEMV(X_tensor_list, args['layers'], device, use_wavelet=True,
                           wavelet_K=args.get('wavelet_K', 10)).to(device)

    for i, (target_wf, source_wf) in enumerate(
            zip(joint_model.wavelet_filters, trained_wavelet_model.wavelet_filters)):
        target_wf.wavelet_coeffs.data = source_wf.wavelet_coeffs.data.clone()

    all_params = joint_model.get_gae_parameters() + joint_model.get_wavelet_parameters()
    joint_optimizer = torch.optim.Adam(all_params, lr=args["learning_rate"])

    neighbor_num = args["neighbor_init"]
    weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
        X_tensor_list, neighbor_num)

    # 预训练
    for epoch in range(args["pretrain_epoch"]):
        for i in range(args["pretrain_iter"]):
            wavelet_filter_matrices = []
            for j, (eigenvals, eigenvecs, wavelet_filter) in enumerate(
                    zip(eigenvals_list, eigenvecs_list, joint_model.wavelet_filters)):
                filter_matrix = wavelet_filter.forward_with_eigen(eigenvals, eigenvecs)
                wavelet_filter_matrices.append(filter_matrix)

            embedding_list, recons_w_list = joint_model(X_tensor_list, wavelet_filter_matrices)
            re_loss, tr_loss = joint_model.cal_loss(
                raw_weights_mv, recons_w_list, weights_mv, embedding_list, args["lam_tr"]
            )
            spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                          joint_model.wavelet_filters)
            total_loss = re_loss + args["lam_tr"] * tr_loss + args.get("beta", 0.1) * spectral_loss

            joint_optimizer.zero_grad()
            total_loss.backward()
            joint_optimizer.step()

        weights_mv, raw_weights_mv, target_filters, eigenvals_list, eigenvecs_list = update_graph_with_eigen_cache(
            embedding_list, neighbor_num)
        if neighbor_num < args["neighbor_max"]:
            neighbor_num = min(neighbor_num + args["neighbor_incr"], args["neighbor_max"])

    # 微调
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
        con_loss = 0
        for vi in range(n_view):
            for vj in range(vi + 1, n_view):
                con_loss += mse_loss_func(embedding_list[vi], embedding_list[vj])
        spectral_loss = compute_spectral_fitting_loss(eigenvals_list, eigenvecs_list, target_filters,
                                                      joint_model.wavelet_filters)
        total_loss = re_loss + args["lam_tr"] * tr_loss + args["lam_con"] * con_loss + args.get("beta", 0.1) * spectral_loss

        joint_optimizer.zero_grad()
        total_loss.backward()
        joint_optimizer.step()

        if (epoch + 1) % args["log_freq"] == 0:
            print(f'Epoch[{epoch + 1}/{args["finetune_epoch"]}], Total_L: {total_loss.item():.4f}')

    print("\n=== Evaluating After Joint Stage ===")
    nmi, nmi_std, acc, acc_std, f1, f1_std = evaluate_model(joint_model, embedding_list, target_filters, eigenvals_list,
                                                            eigenvecs_list, Y, n_cluster, args)
    joint_results = (acc, acc_std, nmi, nmi_std, f1, f1_std)
    print(f"Joint Stage - ACC: {acc:.2f}, NMI: {nmi:.2f}, F1: {f1:.2f}")
    return joint_model, joint_results

def spectral_fitting_experiment(dataset_name, args):
    print(f"\n{'=' * 80}\nSPECTRAL Fitting EXPERIMENT: {dataset_name}\n{'=' * 80}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_list, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)
    X_tensor_list = []
    for i, X in enumerate(X_list):
        if scipy.sparse.issparse(X):
            X_dense = X.toarray()
            X_tensor = torch.from_numpy(X_dense).float().to(device)
        else:
            X_tensor = torch.from_numpy(np.array(X)).float().to(device)
        X_tensor_list.append(X_tensor)

    trained_wavelet_model, fitting_results = train_spectral_fitting_stage(
        X_tensor_list, Y, n_cluster, n_sample, n_view, args, device
    )
    joint_model, joint_results = train_joint_finetuning_stage(
        X_tensor_list, Y, n_cluster, n_sample, n_view, trained_wavelet_model, args, device
    )

    dist_acc, _, dist_nmi, _, dist_f1, _ = fitting_results
    joint_acc, _, joint_nmi, _, joint_f1, _ = joint_results

    return {
        'spectral_fitting': {'acc': dist_acc, 'nmi': dist_nmi, 'f1': dist_f1},
        'joint_finetuning': {'acc': joint_acc, 'nmi': joint_nmi, 'f1': joint_f1},
        'framework_quality': "SUPERIOR" if (joint_acc - dist_acc) > 1 else "COMPARABLE"
    }