import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, cohen_kappa_score


def CalculatePrecisionRecallF1Metrics(_abnormal_label, _y_pred):
    precision = precision_score(_abnormal_label, _y_pred)
    recall = recall_score(_abnormal_label, _y_pred)
    f1 = f1_score(_abnormal_label, _y_pred)
    return precision, recall, f1


def CreateTopKLabelBasedOnReconstructionError(_error, _k):
    label = np.full(_error.shape[0], 1)
    outlier_indices = _error.argsort()[-_k:][::-1]
    for i in outlier_indices:
        label[i] = -1
    return label, outlier_indices


def CalculatePrecisionAtK(_abnormal_label, _score, _k, _type):
    y_pred_at_k = np.full(_k, -1)
    if _type == 1:  # Local Outlier Factor & Auto-Encoder Type
        # _score[_score > 2.2] = 1
        outlier_indices = _score.argsort()[-_k:][::-1]
    if _type == 2:  # Isolation Forest & One-class SVM Type
        outlier_indices = _score.argsort()[:_k]
    abnormal_at_k = []
    for i in outlier_indices:
        abnormal_at_k.append(_abnormal_label[i])
    abnormal_at_k = np.asarray(abnormal_at_k)
    precision_at_k = precision_score(abnormal_at_k, y_pred_at_k)
    return precision_at_k


def CalculateROCAUCMetrics(_abnormal_label, _score):
    fpr, tpr, _ = roc_curve(_abnormal_label, _score)
    roc_auc = auc(np.nan_to_num(fpr), np.nan_to_num(tpr))
    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc
    return fpr, tpr, roc_auc


def CalculateCohenKappaMetrics(_abnormal_label, _y_pred):
    cks = cohen_kappa_score(_abnormal_label, _y_pred)
    if cks < 0:
        cks = 0
    return cks


def CalculatePrecisionRecallCurve(_abnormal_label, _score):
    precision_curve, recall_curve, _ = precision_recall_curve(_abnormal_label, _score)
    average_precision = average_precision_score(_abnormal_label, _score)
    if average_precision < 0.5:
        average_precision = 1 - average_precision
    return precision_curve, recall_curve, average_precision


def CalculateFinalAnomalyScore(_ensemble_score):
    final_score = np.median(_ensemble_score, axis=0)
    return final_score


def PrintPrecisionRecallF1Metrics(_precision, _recall, _f1):
    print('precision=' + str(_precision))
    print('recall=' + str(_recall))
    print('f1=' + str(_f1))


def CalculateAverageMetric(_sum_of_score):
    '''
    Calculate average score of a set of multiple dataset
    :param _sum_of_score: Python List [] of score
    :return: average score
    '''
    average_score = sum(_sum_of_score) / float(len(_sum_of_score))
    return average_score


def PrintROCAUCMetrics(_fpr, _tpr, _roc_auc):
    print('fpr=' + str(_fpr))
    print('tpr=' + str(_tpr))
    print('roc_auc' + str(_roc_auc))


def SquareErrorDataPoints(_input, _output):
    input = np.squeeze(_input, axis=0)
    output = np.squeeze(_output, axis=0)
    # Caculate error
    error = np.square(input - output)
    error = np.sum(error, axis=1)
    return error


def Z_Score(_error):
    mu = np.nanmean(_error)
    gamma = np.nanstd(_error)
    zscore = (_error - mu)/gamma
    return zscore


def PlotResult(_values):
    plt.plot(_values)
    plt.show()
