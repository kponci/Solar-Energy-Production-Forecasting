import datetime

import pandas as pd


def preprocess_train_data(X: pd.DataFrame, y: pd.DataFrame, *args: [str]):
    """
    Usage: preprocess_data(X, y, "everything")
    :param X: features, data from either X_train_observed.parquet or X_train_estimated
    :param y: targets, data from train_targets.parquet
    :param args: how do you want to modify X, y
    :return: preprocessed X, y
    """
    if "everything" in args:
        args = ["align times", "drop 'times' column in y", "remove 'date calc' column", "split 'date forecast'",
                "replace NaNs for zeros"]
    if "align times" in args:
        # several times misalignments, including:
        #       different starting and ending time
        #       y is energy production every hour, but X is weather every 15 minutes
        X = X[X['date_forecast'].isin(y['time'])].reset_index(drop=True)
        y = y[y['time'].isin(X['date_forecast'])].reset_index(drop=True)
        assert X.shape[0] == y.shape[0], f"X.shape = {X.shape}, y.shape = {y.shape}"
    if "drop 'times' column in y" in args:
        # we want to predict power usage, no need to have timestamp in "y"
        y.drop('time', axis=1, inplace=True)
    if "remove 'date calc' column" in args:
        # IMO date of forecast calculation is irrelevant for our predictions
        if 'date_calc' in X:
            X.drop(labels='date_calc', axis=1, inplace=True)
    if "split 'date forecast'":
        # split date forecast into more relevant features - day of the year and time of the day
        X['day_of_year'] = X['date_forecast'].apply(lambda x: x.dayofyear)
        X['hour_of_day'] = X['date_forecast'].apply(lambda x: x.hour)
        X.drop('date_forecast', axis=1, inplace=True)
    if "replace NaNs for zeros":
        # there are NaNs in X (e.g. feature "snow_density"), replace them for zeros
        X.fillna(0, inplace=True)
    return X, y


def preprocess_test_data(X: pd.DataFrame, *args: [str]):
    """
    Usage: preprocess_test_data(X, "everything")
    :param X: features, data from X_train_observed.parquet concatenated with X_train_estimated
    :param args: how do you want to modify X
    :return: preprocessed X
    """
    # function similar to "preprocess_train_data", but it does not receive y
    start_time = X['date_forecast'].iloc[0]
    end_time = X['date_forecast'].iloc[-1] + datetime.timedelta(hours=1)
    time_range = pd.date_range(start=start_time, end=end_time, freq="1H")
    y_temp = pd.DataFrame({'time': time_range, 'pv_measurement': 0})

    X, _ = preprocess_train_data(X, y_temp, *args)
    return X
