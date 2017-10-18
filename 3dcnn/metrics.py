import sklearn
import numpy as np
from sklearn.metrics import f1_score

def metric(y_preds, y_trues):

    y_pred = np.concatenate([y_p for y_p in y_preds])
    y_true = np.concatenate([y_t for y_t in y_trues])
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    f1_score_ = f1_score(y_true, y_pred, average='macro')
    return f1_score_


def metric_(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    f1_score_ = f1_score(y_true, y_pred, average='macro')
    return f1_score_