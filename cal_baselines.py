import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# a = np.random.normal(0, 3, 1000)
# b = np.random.normal(2, 4, 900)
# mpl.use('TkAgg')

path_list = [
    # ######## calculate resuls for SecMI #############
    # r"inter_output/SecMI/Atk_sec_M_coco_real_split1_DATA_val17_split1__TRTE_train_T_0510_172145.txt",
    # r"inter_output/SecMI/Atk_sec_M_coco_real_split1_DATA_val17_split1__TRTE_test_T_0510_172145.txt",
    # r"inter_output/SecMI/Atk_sec_M_coco_real_ori_DATA_val17_TRTE_train_T_0510_172145.txt",
    # r"inter_output/SecMI/Atk_sec_M_coco_real_ori_DATA_val17_TRTE_test_T_0510_172145.txt",

    ######## calculate resuls for PIA #############
    # r"inter_output/PIA/Atk_pia_M_coco_real_split1_DATA_val17_split1__TRTE_train_T_0510_172145.txt",
    #     r"inter_output/PIA/Atk_pia_M_coco_real_split1_DATA_val17_split1__TRTE_test_T_0510_172145.txt",
    #     r"inter_output/PIA/Atk_pia_M_coco_real_ori_DATA_val17_TRTE_train_T_0510_172145.txt",
    #     r"inter_output/PIA/Atk_pia_M_coco_real_ori_DATA_val17_TRTE_test_T_0510_172145.txt"

    ########### calculate resuls for PFAMI ##############
    r"inter_output/PFAMI/Atk_fluc_M_coco_real_split1_DATA_val17_split1__TRTE_train_T_0518_160926.txt",
    r"inter_output/PFAMI/Atk_fluc_M_coco_real_split1_DATA_val17_split1__TRTE_test_T_0518_160926.txt",
    r"inter_output/PFAMI/Atk_fluc_M_coco_real_ori_DATA_val17_TRTE_train_T_0518_160926.txt",
    r"inter_output/PFAMI/Atk_fluc_M_coco_real_ori_DATA_val17_TRTE_test_T_0518_160926.txt",

]

MetricName = '---'
Dataname = '---'


def get_ori_data(path_train, path_test, use_fluc):
    if use_fluc == False:
        with open(path_train, 'r', encoding='utf8') as f:
            train_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

        with open(path_test, 'r', encoding='utf8') as f:
            test_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]
    else:
        with open(path_train, 'r', encoding='utf8') as f:
            train_list = [[-float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

        with open(path_test, 'r', encoding='utf8') as f:
            test_list = [[-float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    ####
    train = np.array(train_list)
    test = np.array(test_list)
    # print("train.shape, test.shape:", train.shape, test.shape)

    # max_v = max(train.max(), test.max())
    # # max_v = max(.max(), sorted(test[2:-1],key= lambda x:x[0]).max())
    # print("max_v", max_v)
    # min_v = min(train.min(), test.min())
    # print("min_v", min_v)

    return train, test,  # max_v, min_v





def deal_data_first(train, test):
    global MetricName
    MetricName = 'first_cond'
    train_ratio = [e[0] for e in train]
    test_ratio = [e[0] for e in test]

    return np.array(train_ratio), np.array(test_ratio)



#####
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def get_th(train, test, n_points=2000):
    best_threshold = None
    best_accuracy = 0.0

    labels = [0] * len(train) + [1] * len(test)
    datas = np.concatenate((train, test))

    assert len(labels) == datas.shape[0]

    min_threshold = min(datas)
    max_threshold = max(datas)
    threshold_step = (max_threshold - min_threshold) / 2000

    for threshold in list(np.arange(min_threshold, max_threshold, threshold_step)):

        predicted_values = [1 if value > threshold else 0 for value in datas]

        accuracy = accuracy_score(labels, predicted_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # print('|   best_accuracy, best_threshold, th% :', best_accuracy, best_threshold,
    #       (best_threshold - min_threshold) / (max_threshold - min_threshold))

    auc = roc_auc_score(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    # print("|    AUC Score:", auc)

    fpr, tpr, _ = roc_curve(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    th_percent = (best_threshold - min_threshold) / (max_threshold - min_threshold)

    return best_threshold, best_accuracy, auc, fpr, tpr, tpr_at_1_percent_fpr, th_percent


def get_cls_withTh(train, test, th):
    train_list = train
    test_list = test

    TP = (train_list <= th).sum()
    TN = (test_list > th).sum()
    FP = (test_list <= th).sum()
    FN = (train_list > th).sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ASR = (TP + TN) / (TP + TN + FP + FN)

    return ASR  # best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e


def draw_distribute_auc(train, test, best_threshold, best_asr, auc, FPR_list, TPR_list, th_pred=None,
                        asr_pred=None, tpr_at_1_percent_fpr=None):
    '''

    :param train:  numpy
    :param test:  numpy
    :param best_threshold:
    :param best_asr:
    :param auc:
    :param FPR_list:
    :param TPR_list:
    :param th_pred:
    :param asr_pred:
    :param tpr_at_1_percent_fpr:
    :return:
    '''
    print('draw_distribute_auc tpr_at_1_percent_fpr:', tpr_at_1_percent_fpr)

    train, test = np.array(train), np.array(test)

    max_e, min_e = max(max(train), max(test)), min(min(train), min(test))

    # a = np.array(train_list)
    # b = np.array(test_list)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    bins = np.linspace(min_e, max_e, 200)

    # plt.figure()
    axs[0].hist(train, bins, alpha=0.5, label='Train data')
    axs[0].hist(test, bins, alpha=0.5, label='Test data')
    axs[0].legend(loc='upper left', )
    axs[0].axvline(x=best_threshold, color='r', linestyle='--')
    print('th_pred:', th_pred)
    if th_pred != None:
        axs[0].axvline(x=th_pred, color='blue', linestyle='--')
        print('th_pred 2:', th_pred)
        title_str = 'TrueAsr {:.4f}, PredAsr {:.4f}; TrueTh {:.3} Perc {:.3f}, PredTh {:.3}'.format(
            best_asr,
            asr_pred,
            best_threshold,
            (best_threshold - min_e) / (max_e - min_e),
            th_pred,
        )
        print('title_str', title_str)
    else:
        title_str = 'TrueAsr {:.3f}, TrueTh {:.3}, Perc {:.3f}'.format(
            best_asr,
            best_threshold,
            (best_threshold - min_e) / (max_e - min_e)
        )
    axs[0].set_title(title_str)

    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(FPR_list) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = TPR_list[idx_1_percent_fpr]
    plt.scatter(FPR_list[idx_1_percent_fpr], TPR_list[idx_1_percent_fpr], marker='o', color='red',
                label='1%% FPR (TPR = %0.4f)' % tpr_at_1_percent_fpr)

    axs[1].plot(FPR_list, TPR_list, 'k--',
                label='ROC {0:.4f}'.format(auc, ), lw=2)
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')  # 可
    axs[1].set_title('ROC Curve')
    axs[1].legend(loc="lower right")

    plt.tight_layout()

    print("\nDataName [{}], MetricName [{}]\n".format(Dataname, MetricName))

    plt.show()


if __name__ == '__main__':

    deal_data = deal_data_first

    print('begin ')

    print("***** get shadow data *********")
    path_shadow_train, path_shadow_test, \
    path_target_train, path_target_test = path_list[0], path_list[1], path_list[2], path_list[3]

    use_fluc = False
    if 'fluc' in path_list[0] and 'fluc' in path_list[1] and 'fluc' in path_list[2] and 'fluc' in path_list[3]:
        print(' ----------------- use fluc ')
        use_fluc = True

    shadow_train, shadow_test = get_ori_data(path_shadow_train, path_shadow_test, use_fluc)

    shadow_train, shadow_test = deal_data(shadow_train, shadow_test)

    th_shadow, asr_shadow, auc_shadow, fpr_shadow, tpr_shadow, \
    tpr_at_1_percent_fpr_shadow, th_percent_shadow = get_th(shadow_train, shadow_test)

    print("**** Shadow: ****", )
    print(f"th_shadow {th_shadow} percent {th_percent_shadow}, asr_shadow {asr_shadow},",
          f"auc_shadow {auc_shadow}, tpr_at_1_percent_fpr_shadow {tpr_at_1_percent_fpr_shadow}\n")

    print("***** get taget data *********")
    target_train, target_test, = get_ori_data(path_target_train,
                                              path_target_test,
                                              use_fluc)  # (path1, path2)#     (path_test1, path_test2)#
    target_train, target_test = deal_data(target_train, target_test, )

    print('*** cal max_asr of target model ***')

    _th_target, _asr_target, auc_target, fpr_target, tpr_target, \
    tpr_at_1_percent_fpr_target, _th_percent_target = get_th(target_train, target_test)

    print(f"Max: (th_target {_th_target} percent {_th_percent_target}, asr_target {_asr_target})\n",
          f"auc_target {auc_target},  tpr_at_1_percent_fpr_target {tpr_at_1_percent_fpr_target}\n")

    print('*** TEST target using shadow TH ***')
    asr_pred = get_cls_withTh(target_train, target_test, th=th_shadow)
    print('Pre-TEST: ', 'ASR_pred:', asr_pred, 'by the given threshold:', th_shadow)

    draw_distribute_auc(target_train, target_test, _th_target, _asr_target, auc_target, fpr_target, tpr_target,
                        th_pred=th_shadow, asr_pred=asr_pred, tpr_at_1_percent_fpr=tpr_at_1_percent_fpr_target)
