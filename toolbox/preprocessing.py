import pandas as pd
from netCDF4 import Dataset
import numpy as np
from opendrift.models.physics_methods import distance_along_trajectory
from datetime import timedelta
from pprint import pprint

# Distance to coastline dataset that you can download here https://pae-paha.pacioos.hawaii.edu/thredds/dist2coast.html?dataset=dist2coast_1deg
D2C = "DATA/FOR_LENNY/dist2coast_1deg.nc"


timestep = {
    "25h": 90000,
    "1j": 86400,
    "12h": 43200,
    "6h": 21600,
    "3h": 10800,
    "1h": 3600,
    "30min": 1800,
    "15min": 900,
    "10min": 600,
}


class Nearest2DInterpolator:

    def __init__(self, xgrid, ygrid, x, y):
        self.x = x
        self.y = y
        self.xi = (x - xgrid.min()) / (xgrid.max() - xgrid.min()) * len(xgrid)
        self.yi = (y - ygrid.min()) / (ygrid.max() - ygrid.min()) * len(ygrid)
        self.xi = np.round(self.xi).astype(np.uint32)
        self.yi = np.round(self.yi).astype(np.uint32)
        self.xi[self.xi >= len(xgrid)] = len(xgrid) - 1
        self.yi[self.yi >= len(ygrid)] = len(ygrid) - 1

    def __call__(self, array2d):
        return array2d[self.yi, self.xi]


def formatting(
    filename: str, column_names: dict, date_format: str, No_column=False
) -> pd.DataFrame:
    """Find the columns Latitude, Longitude and date in the ccsv file thanks to the dictionnary column_names.
    Convert the time data with the date_format.
    Sorts by date, drops duplicates and converts the longitudes in 0-360

    Args:
        filename (str): path of the csv file
        column_names (dict): column names correspondance
        date_format (str): string format according to datetime format
        No_column (bool, optional): Set to True if there is no column name in the csv file. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    if No_column:
        data = pd.read_csv(filename, sep=",", header=None)
        data = data.rename(columns=column_names)                 #hardcoded to column_names!
    else:
        data = pd.read_csv(filename, sep=",")
        data = data.rename(columns=column_names)
    new_col_names = [column_names[k] for k in column_names.keys()]
    data = data[new_col_names]
    data["time"] = pd.to_datetime(data["time"], format=date_format)
    data = data.sort_values(by=["time"])
    data = data.drop_duplicates(subset=["time"])
    data["Longitude"] = data["Longitude"].values % 360
    return data


def filt_timegap(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Add a column to labelise the data to be cut later in the preprocessing function

    Args:
        data (pd.DataFrame): _description_
        threshold (int): maximum number of days of a time gap

    Returns:
        pd.DataFrame: _description_
    """
    dt = data["time"].diff()
    data.loc[dt.dt.days < threshold, "cut_time"] = 0
    data.loc[0, "cut_time"] = 0
    return data


def Dist2CoastLine(x, y, D2C_dataset: str = D2C):
    """Compute the distance to coast line of each point

    Args:
        x (_type_): longitudes
        y (_type_): latitudes
        D2C_dataset (str): path to Distance to coastline dataset

    Returns:
        list : distances
    """
    Dist = Dataset(D2C_dataset)
    dist = Dist["dist"][:5001].data
    lon = Dist["lon"][:].data
    lat = Dist["lat"][:5001].data
    interp = Nearest2DInterpolator(
        lon,
        lat,
        ((x + 180) % 360) - 180,
        y,
    )
    dist = interp(dist[::-1, ::])
    Dist.close()
    return dist


def filt_dist(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Add a column to labelise the data to be cut later in the preprocessing function

    Args:
        data (pd.DataFrame): _description_
        threshold (int): minimum distance from the coast in km

    Returns:
        pd.DataFrame: _description_
    """

    dist = Dist2CoastLine(
        data.loc[:, "Longitude"].values, data.loc[:, "Latitude"].values, D2C
    )
    data.loc[dist > threshold, "cut_dist"] = 0
    # data.loc[:, "dist"] = dist

    return data


def cut_data(data: pd.DataFrame, minimal_duration: int = 86400) -> list[pd.DataFrame]:
    """Cut the dataframe in dataframes according to cutting label computed previously by filt_dist and filt_time

    Args:
        data (pd.DataFrame): _description_
        minimal_duration (int, optional): Minimal duration of a sub dataframe in seconds. Defaults to 86400.

    Returns:
        pd.DataFrame: _description_
    """

    data.loc[:, "cut"] = data.loc[:, ["cut_time", "cut_dist"]].sum(axis=1, skipna=False)
    data = data.drop(columns=["cut_time", "cut_dist"])
    data = data.reset_index(drop=True)
    valid_indices = data[data["cut"].notna()].index
    unvalid_indices = data[np.logical_not(data["cut"].notna())].index

    dfs = []
    # Split DataFrame based on gaps
    end_idx = 0
    for start_idx in valid_indices:
        if start_idx < end_idx:
            continue
        end_idx = unvalid_indices[unvalid_indices > start_idx]
        if len(end_idx) > 0:
            end_idx = end_idx[0]
        else:
            end_idx = None
        dfi = data.iloc[start_idx:end_idx, :3]
        start = dfi.loc[:, "time"].min()
        end = dfi.loc[:, "time"].max()
        if (end - start).total_seconds() >= minimal_duration:
            dfs.append(dfi)
        if end_idx is None:
            break

    return dfs


def interpolation(
    data: pd.DataFrame, ts: int, velocity_threshold: int = 10
) -> pd.DataFrame:
    """Interpolates the dataframe to a desired timestep and remove points where velocity is above a threshold

    Args:
        data (pd.DataFrame): _description_
        ts (int): timestep in second
        velocity_threshold (int, optional): maximum velocity allowed. Defaults to 10.

    Returns:
        pd.DataFrame: _description_
    """
    dt = data["time"].diff().dt.total_seconds()
    data.index = pd.to_datetime(data.loc[:, "time"])
    data = data.drop(columns="time")
    data["Longitude"] = np.rad2deg(np.unwrap(np.deg2rad(data["Longitude"])))
    dists = distance_along_trajectory(data.loc[:, "Longitude"], data.loc[:, "Latitude"])
    vel = dists / dt.values[1:]
    data["Longitude"][1:][vel > velocity_threshold] = np.nan

    data_resampled = data.resample(
        timedelta(seconds=60)
    ).mean()  # Resampling to min intervals and taking the mean
    # Interpolate missing values (linear interpolation)
    df_interpolated = data_resampled.interpolate(method="linear")
    data_resampled = df_interpolated.resample(timedelta(seconds=ts)).mean()
    data_resampled["Longitude"] += 360
    data_resampled["Longitude"] %= 360
    return data_resampled


def preprocessing(
    filename,
    column_names,
    date_format,
    time_thresh,
    dist_thresh,
    timestep_interpolation,
    minimal_duration=timestep["1j"],
    velocity_thresh=10,
    No_column=False,
    Cut=True,
):
    """Apply the different functions above one by one to preprocess a csv file

    Args:
        filename (_type_): path of the csv file
        column_names (_type_): dictionnary to explain which column corresponds to what. Latitude , Longitude and Time
        date_format (_type_): what is the date format of the data in th ecsv file ?
        time_thresh (_type_): threshold to apply to the data. 2 points separated by a time longer than the threshold will be separated in two different dataframes
        dist_thresh (_type_): threshold to apply to the data. A point having a distance to the coastline smaller than the threshold will removed and the data will be cut in two different dataframes
        timestep_interpolation (_type_): desired timestep to interpolate the observation: a timestep of 1 day will generate data with one point every day etc...
        minimal_duration (_type_, optional): minimal duration of an observation under what we don't use the data. Defaults to timestep["1j"].
        velocity_thresh (int, optional): maximum velocity to remove outliers. Defaults to 10.
        No_column (bool, optional): if the initial csv file doesn't have column names, set to True. Defaults to False.

    Returns:
        _type_: list of dataframe
    """

    data = formatting(
        filename,
        column_names=column_names,
        date_format=date_format,
        No_column=No_column,
    )
    if any(np.logical_and(data["Latitude"].values > 65, data["Latitude"].values < 68)):
        pass
    if Cut:
        data = filt_timegap(data, time_thresh)
        data = filt_dist(data, dist_thresh)
        dfs = cut_data(data, minimal_duration)
    else:
        dfs = [data]
    if not timestep_interpolation is None:
        dfs = [interpolation(df, timestep_interpolation, velocity_thresh) for df in dfs]

    if len(dfs) == 1:
        print("No cut in the file")
        return dfs
    else:
        print(f"the file has been cutted in {len(dfs)} subset")
        return dfs
