import os

if __name__ == "__main__":
    # normal
    base_config_trn = "python train_1st_order.py --gpu_id 0 --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA "
    vanilla_config_trn = base_config_trn + "--uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true "
    weaken_config_trn = base_config_trn + "--uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_weaken_02 --weaken 0.2 "

    base_config_val = "python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --SOTA"
    vanilla_config_val = base_config_val + " --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/ "
    weaken_config_val = base_config_val + " --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_weaken_02/ --weaken 0.2 " 
    # 👆后来发现这个weaken参数在验证时不起作用，也就是说在训练时有weaken，在推理时没有weaken

    base_config_plot = "python plot.py "
    vanilla_config_plot = base_config_plot + "--path ./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/ "
    weaken_config_plot = base_config_plot + "--path ./model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true_weaken_02/ "

    # os.system(weaken_config_trn)
    # os.system(weaken_config_val)
    # os.system(weaken_config_plot)
    #
    # os.system(weaken_config_trn)
    # os.system(weaken_config_val)
    # os.system(weaken_config_plot)

    # ETF fc
    vanilla_etf_config_trn = vanilla_config_trn.replace("ETFfc_false", "ETFfc_true") + " --ETF_fc"
    weaken_etf_config_trn = weaken_config_trn.replace("ETFfc_false", "ETFfc_true") + " --ETF_fc"

    vanilla_etf_config_val = vanilla_config_val.replace("ETFfc_false", "ETFfc_true") + " --ETF_fc"
    weaken_etf_config_val = weaken_config_val.replace("ETFfc_false", "ETFfc_true") + " --ETF_fc"

    vanilla_etf_config_plot = vanilla_config_plot.replace("ETFfc_false", "ETFfc_true")
    weaken_etf_config_plot = weaken_config_plot.replace("ETFfc_false", "ETFfc_true")

    # os.system(vanilla_etf_config_trn)
    # os.system(vanilla_etf_config_val)
    # os.system(vanilla_etf_config_plot)
    #
    # os.system(weaken_etf_config_trn)
    # os.system(weaken_etf_config_val)
    # os.system(weaken_etf_config_plot)

    # fixdim to: 10
    vanilla_fixdim_config_trn = vanilla_config_trn.replace("fixdim_false", "fixdim_10") + " --fixdim 10"
    weaken_fixdim_config_trn = weaken_config_trn.replace("fixdim_false", "fixdim_10") + " --fixdim 10"

    vanilla_fixdim_config_val = vanilla_config_val.replace("fixdim_false", "fixdim_10") + " --fixdim 10"
    weaken_fixdim_config_val = weaken_config_val.replace("fixdim_false", "fixdim_10") + " --fixdim 10"

    vanilla_fixdim_config_plot = vanilla_config_plot.replace("fixdim_false", "fixdim_10")
    weaken_fixdim_config_plot = weaken_config_plot.replace("fixdim_false", "fixdim_10")

    os.system(vanilla_fixdim_config_trn)
    os.system(vanilla_fixdim_config_val)
    os.system(vanilla_fixdim_config_plot)

    os.system(weaken_fixdim_config_trn)
    os.system(weaken_fixdim_config_val)
    os.system(weaken_fixdim_config_plot)

    # fixdim to 10 and ETF fc
    vanilla_etf_fixdim_config_trn = vanilla_config_trn.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10") + " --ETF_fc --fixdim 10"
    weaken_etf_fixdim_config_trn = weaken_config_trn.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10") + " --ETF_fc --fixdim 10"

    vanilla_etf_fixdim_config_val = vanilla_config_val.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10") + " --ETF_fc --fixdim 10"
    weaken_etf_fixdim_config_val = weaken_config_val.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10") + " --ETF_fc --fixdim 10"

    vanilla_etf_fixdim_config_plot = vanilla_config_plot.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10")
    weaken_etf_fixdim_config_plot = weaken_config_plot.replace("ETFfc_false_fixdim_false", "ETFfc_true_fixdim_10")

    os.system(vanilla_etf_fixdim_config_trn)
    os.system(vanilla_etf_fixdim_config_val)
    os.system(vanilla_etf_fixdim_config_plot)

    os.system(weaken_etf_fixdim_config_trn)
    os.system(weaken_etf_fixdim_config_val)
    os.system(weaken_etf_fixdim_config_plot)
