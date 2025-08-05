import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, insample):
    scale = np.mean(np.abs(np.diff(insample)))
    return np.mean(np.abs(y_true - y_pred)) / scale if scale else np.inf


def skill_score(y_true, y_pred, y_naive):
    rm = np.sqrt(mean_squared_error(y_true, y_pred))
    rn = np.sqrt(mean_squared_error(y_true, y_naive))
    return 1 - (rm / rn) if rn else -np.inf