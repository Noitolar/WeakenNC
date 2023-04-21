import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import os


def plot(lst1, lst2=None, label1=None, label2=None, num_interpolations=0, title="", saveto=None):
    plt.figure(figsize=(12, 9))
    plt.title(title)
    if label1 is None:
        label1 = "vanilla"
    if label2 is None:
        label2 = "weaken"
    plt.plot(*smooth(np.array(lst1), num_interpolations), label=label1, linewidth=3, color="#86cabf")
    if lst2 is not None:
        plt.plot(*smooth(np.array(lst2), num_interpolations), label=label2, linewidth=3, color="#fa8e7a")
    plt.grid()
    plt.legend()
    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()
    plt.close()


def smooth(lst, num_interpolations):
    x = np.linspace(1, len(lst), len(lst))
    y = np.array(lst)
    if num_interpolations <= 0:
        return x, y
    x_smooth = np.linspace(min(x), max(x), num_interpolations)
    smoothing_spline = interpolate.make_smoothing_spline(x, y)
    y_smooth = smoothing_spline(x_smooth)
    return x_smooth, y_smooth


if __name__ == "__main__":

    translator = {
        "collapse_metric": "nc1",
        "ETF_metric": "nc2",
        "WH_relation_metric": "nc3",
        "Wh_b_relation_metric": "nc4",
        "train_acc1": "trn-acc",
        "test_acc1": "val-acc"
    }

    exp_path = {
        "vanilla_normal": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/info.pkl",
        "weaken_normal": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_weaken_02/info.pkl",
        "vanilla_etf": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true/info.pkl",
        "weaken_etf": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true_weaken_02/info.pkl",
        "vanilla_fixdim": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true/info.pkl",
        "weaken_fixdim": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_10_sota_true_weaken_02/info.pkl",
        "vanilla_etf_fixdim": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true/info.pkl",
        "weaken_etf_fixdim": "./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_10_sota_true_weaken_02/info.pkl",
    }

    def xxx(label1, label2, start=0, num_interpolations=0):
        with open(exp_path[label1], 'rb') as f1, open(exp_path[label2], "rb") as f2:
            info1 = pickle.load(f1)
            info2 = pickle.load(f2)
            for key in info1.keys():
                if key in translator.keys():
                    savepath = f"./figures/{label1.replace('vanilla_', '')}/{translator[key]}.png"
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    plot(info1[key][start:], info2[key][start:], label1, label2, num_interpolations, title=translator[key].upper(), saveto=savepath)

    xxx("vanilla_normal", "weaken_normal")
    xxx("vanilla_etf", "weaken_etf")
    xxx("vanilla_fixdim", "weaken_fixdim")
    xxx("vanilla_etf_fixdim", "weaken_etf_fixdim")
