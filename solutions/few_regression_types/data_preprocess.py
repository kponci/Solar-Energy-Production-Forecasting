import pandas as pd


def preprocess_data(X: pd.DataFrame, y: pd.DataFrame, *args: [str]):
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
        # y is energy production every hour, but X is weather every 15 minutes
        # I still need to check how it is in the datasets B and C
        # TODO: implement
        print("ERROR: times will not be aligned!")
        pass
    if "drop 'times' column in y" in args:
        # we want to predict power usage, no need to have timestamp in "y"
        y.drop("time", axis=1, inplace=True)
    if "remove 'date calc' column" in args:
        # IMO date of forecast calculation is irrelevant for our predictions
        X.drop(labels="date_calc", axis=1, inplace=True)
    if "split 'date forecast'":
        # split date forecast into more relevant features - day of the year and time of the day
        X['day_of_year'] = X['date_forecast'].apply(lambda x: x.dayofyear)
        X['hour_of_day'] = X['date_forecast'].apply(lambda x: x.hour)
        X.drop('date_forecast', axis=1, inplace=True)
    if "replace NaNs for zeros":
        # there are NaNs in X (e.g. feature "snow_density"), replace them for zeros
        X.fillna(0, inplace=True)
    return X, y
