from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast(y_train, X_train, X_test, horizon=3):
    mod = ExponentialSmoothing(y_train, trend='add', seasonal=None)
    fit = mod.fit()
    return fit.forecast(horizon)