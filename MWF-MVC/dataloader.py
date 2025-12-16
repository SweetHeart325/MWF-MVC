import os
import scipy.io as sio
import h5py
import numpy as np
import scipy.sparse
from sklearn import preprocessing

def load_new_format_data(dataset_name):
    # 假设 dataset 文件夹在当前脚本的同级目录
    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_dir, 'dataset', f"{dataset_name}.mat")

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

    if "X" in data:
        X = data["X"]
    elif "x" in keymap:
        X = data[keymap["x"]]
    elif "xs" in keymap:
        X = data[keymap["xs"]]
    else:
        raise KeyError(f"'{data_path}' X/x/Xs not found.")

    if "y" in keymap:
        Y = data[keymap["y"]]
    elif "Y" in data:
        Y = data["Y"]
    else:
        raise KeyError(f"'{data_path}' y/Y not found.")

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
    normalized_X = []
    for i, x in enumerate(X):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

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
            except Exception:
                try:
                    scaler = preprocessing.MinMaxScaler()
                    x_normalized = scaler.fit_transform(x)
                    normalized_X.append(x_normalized)
                except Exception:
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