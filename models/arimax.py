import pmdarima as pm

def forecast(y_train, X_train, X_test, horizon=3):
    model = pm.auto_arima(
        y_train, exogenous=X_train,
        seasonal=False, stepwise=True,
        suppress_warnings=True, error_action='ignore'
    )
    return model.predict(n_periods=horizon, exogenous=X_test)