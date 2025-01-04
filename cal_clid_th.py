import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# a = np.random.normal(0, 3, 1000)
# b = np.random.normal(2, 4, 900)
# mpl.use('TkAgg')

############ shadow model result ###############
path1 = [
r"inter_output/CLID/Atk_Impt_M_coco_real_split1_DATA_val17_split1_TRTE_train_MAXsmp_3_T_0506_145909.txt"
][0]

path2 = [
r"inter_output/CLID/Atk_Impt_M_coco_real_split1_DATA_val17_split1__TRTE_test_MAXsmp_3_T_0506_145909.txt"
][0]


############ target model result ###############
path_test1 = [
r"inter_output/CLID/Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_train_MAXsmp_3_T_0506_145842.txt"
][0]

path_test2 = [
r"inter_output/CLID/Atk_Impt_M_coco_real_ori_DATA_val17_TRTE_test_MAXsmp_3_T_0506_145842.txt"

][0]

MetricName = '---'
Dataname = '---'


def get_ori_data(path_train, path_test):
    global Dataname
    if '/' in path_train:
        Dataname = path_train.split('/')[-2]
    elif '\\' in path_train:
        Dataname = path_train.split('\\')[-2]

    print('dataname:', Dataname)
    # exit()

    with open(path_train, 'r', encoding='utf8') as f:
        train_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    with open(path_test, 'r', encoding='utf8') as f:
        test_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    train = np.array(train_list)
    test = np.array(test_list)
    # print("train.shape, test.shape:", train.shape, test.shape)

    # max_v = max(train.max(), test.max())
    # # max_v = max(.max(), sorted(test[2:-1],key= lambda x:x[0]).max())
    # print("max_v", max_v)
    # min_v = min(train.min(), test.min())
    # print("min_v", min_v)

    return train, test,  # max_v, min_v


def get_l_clidavg_last3(train, test):
    # global MetricName
    # MetricName = 'get_l_clidavg'
    train_out = [[e[0], - sum(e[1:]) / len(e[1:])] for e in train]
    test_out = [[e[0], - sum(e[1:]) / len(e[1:])] for e in test]

    return train_out, test_out



def deal_data_weight_avg(train, test, alpha):
    global MetricName
    MetricName = 'weight_avg'
    assert len(train[0]) == 2
    train = [(1 - alpha) * e[0] + alpha * e[1] for e in train]
    test = [(1 - alpha) * e[0] + alpha * e[1] for e in test]

    return np.array(train), np.array(test)





def get_th(train, test, n_points=2000):
    print('get th...')
    train_list = train
    test_list = test
    max_e =  max(np.concatenate((train_list, test_list)))  #  max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))
    # n_points = 2000
    best_asr = 0
    best_threshold = 0

    FPR_list = []
    TPR_list = []
    from sklearn import metrics
    # print("\ntrain_list[:3], test_list[:3]", train_list[:3], test_list[:3])
    # print("\nmax_e, min_e:", max_e, min_e)

    for threshold in list(np.arange(min_e, max_e, (max_e - min_e) / n_points)):
        # print(threshold, type(threshold))
        TP = (train_list <= threshold).sum()
        TN = (test_list > threshold).sum()
        FP = (test_list <= threshold).sum()
        FN = (train_list > threshold).sum()
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        ASR = (TP + TN) / (TP + TN + FP + FN)
        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > best_asr:
            best_asr = ASR
            best_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)

    # print('\n', 'best_asr:', best_asr, 'best_threshold:', best_threshold, 'percent:',
    #       (best_threshold - min_e) / (max_e - min_e), 'AUC:', auc)

    return best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e


def get_cls_withTh(train, test, th):
    train_list = train
    test_list = test
    max_e = max(np.concatenate((train_list, test_list)))  # max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))
    # n_points = 2000

    print("\ntrain_list[:3], test_list[:3]", train_list[:3], test_list[:3])
    print("\nmax_e, min_e:", max_e, min_e)

    TP = (train_list <= th).sum()
    TN = (test_list > th).sum()
    FP = (test_list <= th).sum()
    FN = (train_list > th).sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ASR = (TP + TN) / (TP + TN + FP + FN)

    print('\n', 'TEST: ', 'ASR:', ASR, 'by the given threshold:', th)

    return ASR  # best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e



def get_1_fpr(train_data_target, test_data_target):
    print('**** get get_1_fpr: ******')
    labels = [0]*len(train_data_target)+[1]*len(test_data_target)
    datas = np.concatenate((train_data_target, test_data_target), axis=0)


    best_threshold = None
    best_accuracy = 0.0

    min_threshold = min(datas)
    max_threshold = max(datas)
    threshold_step = (max_threshold - min_threshold) / 2000

    for threshold in list(np.arange(min_threshold, max_threshold, threshold_step)):
        predicted_values = [1 if value > threshold else 0 for value in datas]

        accuracy = accuracy_score(labels, predicted_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print( '|   best_accuracy, best_threshold, th% :', best_accuracy, best_threshold,
          (best_threshold - min_threshold) / (max_threshold - min_threshold))

    auc = roc_auc_score(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    print( "|    AUC Score:", auc)

    fpr, tpr, _ = roc_curve(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    print( "|   tpr_at_1_percent_fpr:", tpr_at_1_percent_fpr)


def draw_distribute_auc(train, test, best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e, th_pred=None,
                        asr_pred=None):
    train_list = train
    test_list = test
    # print(train.shape)
    # print(test.shape)

    a = np.array(train_list).reshape(-1)
    b = np.array(test_list).reshape(-1)

    # print(a.shape, b.shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ###
    max_e = max(np.concatenate((train_list, test_list)))  # max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))

    bins = np.linspace(min_e, max_e, 100)

    # plt.figure()
    axs[0].hist(a, bins, alpha=0.5, label='Train data')
    axs[0].hist(b, bins, alpha=0.5, label='Test data')
    axs[0].legend(loc='upper left', )
    axs[0].axvline(x=best_threshold, color='r', linestyle='--')
    # print('th_pred:', th_pred)
    if th_pred != None:
        axs[0].axvline(x=th_pred, color='blue', linestyle='--')
        # print('th_pred 2:', th_pred)
        # print('\n\n\n---:\n', best_asr,
        #       asr_pred,
        #       best_threshold,
        #       (best_threshold - min_e) / (max_e - min_e),
        #       th_pred, )

        title_str = 'TrueAsr {:.4f}, PredAsr {:.4f}; TrueTh {:.3} Perc {:.3f}, PredTh {:.3}'.format(
            best_asr,
            asr_pred,
            best_threshold,
            (best_threshold - min_e) / (max_e - min_e),
            th_pred,
        )
        print('Prediction title_str', title_str)
    else:
        title_str = 'TrueAsr {:.3f}, TrueTh {:.3}, Perc {:.3f}'.format(
            best_asr,
            best_threshold,
            (best_threshold - min_e) / (max_e - min_e)
        )
    axs[0].set_title(title_str)


    axs[1].plot(FPR_list, TPR_list, 'k--', label='alpha {} ROC = {:.4f}'.format(alpha, auc), lw=2)
    axs[1].set_xlim([-0.05, 1.05])  #
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')  #
    axs[1].set_title('ROC Curve')
    axs[1].legend(loc="lower right")

    plt.tight_layout()

    print("\nDataName [{}], MetricName [{}]\n".format(Dataname, MetricName))



if __name__ == '__main__':
    from sklearn.preprocessing import RobustScaler

    print('begin .. ')
    print('------- get data ------------')
    train_data, test_data, = get_ori_data(path1, path2)

    train_data_2, test_data_2 = get_ori_data(path_test1, path_test2)

    # =
    #### (1-α) x L + α x CLid_avg

    # import random
    #
    # random.shuffle(train_data)
    # random.shuffle(test_data)
    best_alpha = -1
    best_auc_shadow = 0
    test_asr_target = 0
    test_auc_target = 0
    best_train_data_target = None
    best_test_data_target = None

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        train_data_shadow, test_data_shadow = get_l_clidavg_last3(train_data, test_data)

        ########### get scale #############
        Scale = RobustScaler()
        Scale.fit(np.concatenate((train_data_shadow, test_data_shadow)))
        ###########  #############

        train_data_target, test_data_target = get_l_clidavg_last3(train_data_2, test_data_2)

        print(f' **********  alpha [{alpha}] ********** | shadow / target split. \n')

        print("***** deal shadow data *********")
        train_data_shadow, test_data_shadow = Scale.transform(train_data_shadow), Scale.transform(test_data_shadow)
        train_data_shadow, test_data_shadow = deal_data_weight_avg(train_data_shadow, test_data_shadow,
                                                                   alpha)  

        best_threshold_shadow, best_asr_shadow, auc_shadow, FPR_list_sd, TPR_list_sd, max_e_sd, min_e_sd = get_th(
            train_data_shadow, test_data_shadow)  # , max_v, min_v)

        print("**** Shadow: ****", )
        print('Asr:[', best_asr_shadow, ']   best_threshold:', best_threshold_shadow, 'percent:',
              (best_threshold_shadow - min_e_sd) / (max_e_sd - min_e_sd), 'AUC: [', auc_shadow, ']\n',
              'max_e_sd, min_e_sd:', max_e_sd, min_e_sd, '\n\n')


        # draw_distribute_auc(train, test, best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e)

        print("\n***** deal taget data *********")
        train_data_target, test_data_target = Scale.transform(train_data_target), Scale.transform(test_data_target)
        train_data_target, test_data_target = deal_data_weight_avg(train_data_target, test_data_target, alpha)

        # train, test, max_v, min_v = get_ori_data(path_test1, path_test2)#   (path1, path2)#     (path_test1, path_test2)#
        print('*** TEST Model ***')

        best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e = get_th(train_data_target, test_data_target)

        asr_pred = get_cls_withTh(train_data_target, test_data_target, th=best_threshold_shadow)
        print(f'  [{alpha}]   ', 'True best_threshold', best_threshold, 'True best_asr [', best_asr, ']  True AUC [',
              auc, ']', f" ASR pred [{asr_pred}]")

        if auc_shadow>best_auc_shadow:
            best_auc_shadow=auc_shadow
            best_alpha = alpha
            test_asr_target = asr_pred
            test_auc_target = auc
            best_train_data_target = train_data_target
            best_test_data_target = test_data_target



    print('\n cal best_train_data_target, best_test_data_target')
    get_1_fpr(best_train_data_target, best_test_data_target)



    print(f'***********  best_alpha {best_alpha}, best_auc_shadow {best_auc_shadow}, ',  '[test_asr_target], [test_auc_target]:', test_asr_target, test_auc_target)

