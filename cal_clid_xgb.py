import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

    max_v = max(train.max(), test.max())
    # max_v = max(.max(), sorted(test[2:-1],key= lambda x:x[0]).max())
    print("max_v", max_v)
    min_v = min(train.min(), test.min())
    print("min_v", min_v)

    return train, test, max_v, min_v




def deal_data_allconds(train, test):
    global MetricName
    MetricName = 'allconds'
    train = [e for e in train]
    test = [e for e in test]

    return train, test


def get_xgb(train, test, n_estimators=50):
    # GET DATA
    # train_data = np.genfromtxt(path_train, delimiter='\t', skip_header=1)[:,:]
    print('---------Train XGB with n_estimators_{}:\n'.format(n_estimators))

    train = np.array(train)
    test = np.array(test)

    label0 = np.zeros((train.shape[0], 1))
    label1 = np.ones((test.shape[0], 1))

    datas = np.concatenate((train, test))
    print(datas.shape)

    labels = np.concatenate((label0, label1))
    print(labels.shape)

    data_with_label = np.hstack((datas, labels))

    np.random.shuffle(data_with_label)

    # data_with_label
    X = data_with_label[:, :-1]
    y = data_with_label[:, -1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
    X_train = X
    y_train = y

    clf = XGBClassifier(n_estimators=n_estimators,
                        gamma=0.7, max_depth=3, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=1)
    #

    clf.fit(X_train, y_train)

    ## TEST
    y_pred = clf.predict(X_train)
    print(y_pred.shape, y_pred[:5])

    ## CAL asr
    train_acc = np.mean(y_pred == y_train)
    print('Train acc: [{}]'.format(train_acc))

    ## TEST
    # y_pred = clf.predict(X_test)
    # print(y_pred.shape, y_pred[:5])
    #
    # ## CAL asr
    # test_acc = np.mean(y_pred == y_test)
    # print('test acc:', test_acc)
    proba_xgb = clf.predict_proba(X_train)
    print('pred_xgb.shape:', proba_xgb.shape, 'pred_xgb[:, 1].shape', proba_xgb[:, 1].shape)
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y_train, proba_xgb[:, 1])
    print('Train roc_auc: ', auc)

    fpr, tpr, _ = roc_curve(y_train, proba_xgb[:, 1])

    # print('---', y_pred)
    # print('---', proba_xgb)

    # exit()
    return clf, y_pred, y_train, proba_xgb


# @torch.no_grad()
def pre_xgb(train, test, cls):
    print('---------Test XGB on target model:\n')

    train = np.array(train)
    test = np.array(test)

    label0 = np.zeros((train.shape[0], 1))
    label1 = np.ones((test.shape[0], 1))

    datas = np.concatenate((train, test))
    print(datas.shape)

    labels = np.concatenate((label0, label1))
    print(labels.shape)

    data_with_label = np.hstack((datas, labels))

    # np.random.shuffle(data_with_label)

    # data_with_label
    X = data_with_label[:, :-1]
    y = data_with_label[:, -1]


    #
    # clf.fit(X_train, y_train)

    ## TEST
    y_pred = clf.predict(X)
    print(y_pred.shape, y_pred[:5])

    ## CAL asr
    test_acc = np.mean(y_pred == y)
    print('test acc: [{}]'.format(test_acc))

    ## TEST
    # y_pred = clf.predict(X_test)
    # print(y_pred.shape, y_pred[:5])
    #
    # ## CAL asr
    # test_acc = np.mean(y_pred == y_test)
    # print('test acc:', test_acc)
    proba_xgb = clf.predict_proba(X)
    print('pred_xgb.shape:', proba_xgb.shape, 'pred_xgb[:, 1].shape', proba_xgb[:, 1].shape)
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y, proba_xgb[:, 1])
    print('Test roc_auc: ', auc)

    fpr, tpr, _ = roc_curve(y, proba_xgb[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) ASR:{test_acc}')
    plt.legend(loc="lower right")


    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    plt.scatter(fpr[idx_1_percent_fpr], tpr[idx_1_percent_fpr], marker='o', color='red',
                label='1%% FPR (TPR = %0.4f)' % tpr_at_1_percent_fpr)
    plt.legend()

    plt.show()
    print('tpr_at_1_percent_fpr:', tpr_at_1_percent_fpr)

    # exit()
    return clf, y_pred, y, proba_xgb




def custom_format(x):
    return "%.3f" % x


if __name__ == '__main__':
    print("\n**** Shadow: ****", )
    train_sd, test_sd, max_v_sd, min_v_sd = get_ori_data(path1,
                                                         path2)  # (path_test1, path_test2) #(path1, path2)  # (path1, path2)

    train_sd, test_sd = deal_data_allconds(train_sd, test_sd)

    clf, y_pred_sd, y_test_sd, proba_xgb_sd = get_xgb(
        train_sd, test_sd, 20) # 5 10 20 30 40 50 

    print("\n**** Target: ****", )
    train, test, max_v, min_v = get_ori_data(path_test1, path_test2)  # (path1, path2) # (path_test1, path_test2)
    train, test = deal_data_allconds(train, test)
    pre_xgb(train, test, clf)


