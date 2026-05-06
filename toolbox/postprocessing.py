from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import os
from toolbox.preprocessing import preprocessing
import pickle
from opendrift.models.physics_methods import (
    skillscore_liu_weissberg,
    distance_between_trajectories,
)
from pprint import pprint
import matplotlib.pyplot as plt
import pyproj
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from scipy.stats import circmean
from pyproj import Proj, Transformer
from scipy.spatial.distance import pdist
from matplotlib.patheffects import withStroke
from matplotlib.patches import Patch


def compute_IDs(files: list, input_folder: str, outfile: str, prep_params: dict):
    """If the observations are seprated in many different files, this function helps tracking each future seed you will generate for a multiseeding knowing the preprocessing pattern.

    Args:
        files (list): _description_
        input_folder (str): _description_
        outfile (str): _description_
        prep_params (dict): _description_

    Returns:
        _type_: _description_
    """
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            IDs = pickle.load(f)
        return IDs
    else:
        IDs = []
        i = 0
        for file in files:
            fp = os.path.join(input_folder, file)
            with open(fp) as f:
                data = preprocessing(
                    fp,
                    column_names=prep_params["column_names"],
                    date_format=prep_params["date_format"],
                    time_thresh=prep_params["time_thresh"],
                    dist_thresh=prep_params["dist_thresh"],
                    timestep_interpolation=prep_params["ts_interp"],
                    No_column=prep_params["No_column"],
                )
                for df in data:
                    IDs.append(
                        i + len(df.index[:: int(86400 / prep_params["ts_interp"])])
                    )
                    i += len(df.index[:: int(86400 / prep_params["ts_interp"])])
        with open(outfile, "wb") as f:
            pickle.dump(IDs, f)
        return IDs


def match_ID_obs(line: int, IDs: list, files: list):
    wh = np.where(line < np.array(IDs))[0][0]
    # print(wh)
    previous_ID = IDs[wh - 1] if wh > 0 else wh
    real_ID = line - previous_ID
    # print(real_ID)
    return files[wh], real_ID


def create_list_obs(input_folder, prep_params, files=None, outfile=None):
    if outfile is not None:
        if os.path.exists(outfile):
            with open(outfile, "rb") as f:
                list_obs = pickle.load(f)
            return list_obs
    list_obs = []
    if files is None:
        files = os.listdir(input_folder)
    for file in files:
        # print(f"filename : {file}")
        with open(os.path.join(input_folder, file)) as f:
            obs = preprocessing(
                f,
                column_names=prep_params["column_names"],
                date_format=prep_params["date_format"],
                time_thresh=prep_params["time_thresh"],
                dist_thresh=prep_params["dist_thresh"],
                timestep_interpolation=prep_params["ts_interp"],
                No_column=prep_params["No_column"],
            )
        for i in range(len(obs)):
            list_obs.append(obs[i])
    if outfile is not None:
        with open(outfile, "wb") as f:
            pickle.dump(list_obs, f)
    return list_obs


def postprocessing( #This function might be interessting to use
    nc_files,
    input_folder,
    IDs,
    prep_params,
    ts_output: int = 3600,
    files=None,
    outfile=None,
):
    """This function associate a simulation and the corresponding observation for each seed of an opendrift simulation.

    Args:
        nc_files (_type_): _description_
        input_folder (_type_): _description_
        IDs (_type_): _description_
        prep_params (_type_): _description_
        ts_output (int, optional): _description_. Defaults to 3600.
        files (_type_, optional): _description_. Defaults to None.
        outfile (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: An array of matches : a match concatenates the observation and the simulation for the desired duration. Each one concatenates lon,lat.
        Matches = (N Matche,(Observation,simulation),(lon,lat)*duration)
    """
    if not files is None:
        group = list(range(0, len(files), 100))
        group.append(None)
    f = 0
    if not isinstance(nc_files, list):
        nc_files = [nc_files]
    data = Dataset(nc_files[f])                                     # opens first nc file
    mod_time = num2date(data["time"][:], units=data["time"].units) #nice date function
    lon = data["lon"][:].data
    lat = data["lat"][:].data
    previous_seed_number = 0
    new_seed_number = len(lon)
    print(f"{new_seed_number} seeds")
    list_obs = create_list_obs(input_folder, prep_params, files=files, outfile=outfile) #uses pre-processing which is hardcoded to obs column_names!!
    print(len(list_obs))
    # assert len(list_obs) == len(IDs)
    Matches = []
    for i, sub_obs in enumerate(list_obs):
        i_lim0 = 0
        if i > 0:
            i_lim0 = IDs[i - 1]
        i_lim = IDs[i]
        # print(i_lim)
        lim_ID = len(sub_obs)
        # print(lim_ID)
        for j in range(i_lim0, i_lim):
            _, RID = match_ID_obs(j, IDs, list_obs)
            print(f"seed {j} - RID {RID}")
            start_ID_obs = RID * int(86400 / prep_params["ts_interp"])
            end_ID_obs = (
                start_ID_obs + (int(86400 / prep_params["ts_interp"]) + 1)
                if (start_ID_obs + (int(86400 / prep_params["ts_interp"]) + 1) < lim_ID)
                else None
            )

            sub_df = sub_obs.iloc[
                start_ID_obs : end_ID_obs : int(ts_output / prep_params["ts_interp"])
            ]
            # pprint(sub_df)
            if len(sub_df) < int(86400 / ts_output + 1):
                print("Warning : obs too short !")
                continue
            if j - previous_seed_number < new_seed_number:
                indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
            else:
                f += 1
                data = Dataset(nc_files[f])
                mod_time = num2date(data["time"][:], units=data["time"].units)
                lon = data["lon"][:].data
                lat = data["lat"][:].data
                previous_seed_number += new_seed_number
                new_seed_number = len(lon)
                print(f"{new_seed_number} seeds")
                indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
            sub_lon = (lon[j - previous_seed_number, indices_mod] + 360) % 360
            sub_lat = lat[j - previous_seed_number, indices_mod]
            sub_time = mod_time[indices_mod]
            test = np.where(sub_lon < 361)[0]
            if len(test) < int(86400 / ts_output + 1):
                print("Warning : simulation too short !")
                pass
            else:
                print("It's a Match !")
                # reshape lon lat
                sub_lon = np.reshape(sub_lon, (len(sub_lon), 1))
                sub_lat = np.reshape(sub_lat, (len(sub_lon), 1))
                obs_lon = sub_df.loc[:, "Longitude"].values.reshape((len(sub_lon), 1))
                obs_lat = sub_df.loc[:, "Latitude"].values.reshape((len(sub_lon), 1))
                match_obs = np.hstack((obs_lon, obs_lat))
                match_mod = np.hstack((sub_lon, sub_lat))
                _match = np.stack((match_obs, match_mod), axis=0)
                Matches.append(_match)

    if len(Matches) > 0:
        Matches = np.stack(Matches, axis=0)
    # pprint(Matches.shape)
    return Matches


def postprocessing_1(
    nc_file,
    input_folder,
    file,
    IDs,
    prep_params,
    ts_output: int = 3600,
    outfile=None,
    days=1,
):
    with Dataset(nc_file) as data:

        mod_time = num2date(data["time"][:], units=data["time"].units)
        lon = data["lon"][:].data
        lat = data["lat"][:].data
        previous_seed_number = 0
        new_seed_number = len(lon)
        print(f"{new_seed_number} seeds")
        list_obs = create_list_obs(
            input_folder, prep_params, files=file, outfile=outfile
        )
        Matches = []
        for i, sub_obs in enumerate(list_obs):
            i_lim0 = 0
            if i > 0:
                i_lim0 = IDs[i - 1]
            i_lim = IDs[i]
            # print(i_lim)
            lim_ID = len(sub_obs)
            # print(lim_ID)
            for j in range(i_lim0, i_lim):
                _, RID = match_ID_obs(j, IDs, list_obs)
                print(f"seed {j} - RID {RID}")
                start_ID_obs = RID * int(86400 * days / prep_params["ts_interp"])
                end_ID_obs = (
                    start_ID_obs + (int(86400 * days / prep_params["ts_interp"]) + 1)
                    if (
                        start_ID_obs
                        + (int(86400 * days / prep_params["ts_interp"]) + 1)
                        < lim_ID
                    )
                    else None
                )

                sub_df = sub_obs.iloc[
                    start_ID_obs : end_ID_obs : int(
                        ts_output / prep_params["ts_interp"]
                    )
                ]
                # pprint(sub_df)
                if len(sub_df) < int(86400 * days / ts_output + 1):
                    print("Warning : obs too short !")
                    continue
                try:
                    indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
                except IndexError:
                    break
                sub_lon = (lon[j - previous_seed_number, indices_mod] + 360) % 360
                sub_lat = lat[j - previous_seed_number, indices_mod]
                sub_time = mod_time[indices_mod]
                test = np.where(sub_lon < 361)[0]
                if len(test) < int(86400 * days / ts_output + 1):
                    print("Warning : simulation too short !")
                    pass
                else:
                    print("It's a Match !")
                    # reshape lon lat
                    sub_lon = np.reshape(sub_lon, (len(sub_lon), 1))
                    sub_lat = np.reshape(sub_lat, (len(sub_lon), 1))
                    obs_lon = sub_df.loc[:, "Longitude"].values.reshape(
                        (len(sub_lon), 1)
                    )
                    obs_lat = sub_df.loc[:, "Latitude"].values.reshape(
                        (len(sub_lon), 1)
                    )
                    match_obs = np.hstack((obs_lon, obs_lat))
                    match_mod = np.hstack((sub_lon, sub_lat))
                    _match = np.stack((match_obs, match_mod), axis=0)
                    Matches.append(_match)

    if len(Matches) > 0:
        Matches = np.stack(Matches, axis=0)
    return Matches


def compute_SS(matches, outfile, save=False):
    """Compute the skill score and the rmse from a list of matches

    Args:
        matches (_type_): _description_
        outfile (_type_): _description_
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    LON = []
    LAT = []
    SS = []
    RMSE = []
    # plt.hist(matches[:, 0, :, 1].flatten(), bins=100)
    # plt.show()
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]
        ss = skillscore_liu_weissberg(lon_obs, lat_obs, lon_mod, lat_mod)
        dist = distance_between_trajectories(lon_obs, lat_obs, lon_mod, lat_mod)
        rmse = np.sqrt(np.nansum(dist**2) / len(dist))
        lon = obs[0, 0]
        lat = obs[0, 1]
        LON.append(lon)
        LAT.append(lat)
        SS.append(ss)
        RMSE.append(rmse)

    data = {"LON": LON, "LAT": LAT, "SS": SS, "RMSE": RMSE}
    data = pd.DataFrame(data)
    if save:
        data.to_csv(outfile, index=False)
    return data


def statistics(
    data: pd.DataFrame,
    column="SS",
    by=["wind model", "ocean model"],
    outfolder=None,
    **kwargs,
):
    if not os.path.exists(outfolder):
        os.mkdir(os.path.join(".", outfolder))
    if not by is None:
        median = data.loc[:, [column] + by].groupby(by=by).median()
        mean = data.loc[:, [column] + by].groupby(by=by).mean()
        std = data.loc[:, [column] + by].groupby(by=by).std()
        _min = data.loc[:, [column] + by].groupby(by=by).min()
        _max = data.loc[:, [column] + by].groupby(by=by).max()
        combined_df = pd.concat(
            [median, mean, std, _min, _max],
            keys=["median", "mean", "std", "min", "max"],
            axis=1,
        )
        combined_df.to_csv(os.path.join(outfolder, "statistics.txt"), sep="\t")

        ax = data.boxplot(column=column, by=by, **kwargs)

        # Adjust font size
        plt.xticks(fontsize=8)

        # Rotate and align the x-axis labels vertically
        xticklabels = [
            label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(xticklabels)
        plt.savefig(os.path.join(outfolder, "boxplot.png"))
    else:
        median = data.loc[:, column].median()
        mean = data.loc[:, column].mean()
        std = data.loc[:, column].std()
        _min = data.loc[:, column].min()
        _max = data.loc[:, column].max()
        stats = {"median": median, "mean": mean, "std": std, "min": _min, "max": _max}
        stats = pd.DataFrame(stats, index=range(1))
        stats.to_csv(os.path.join(outfolder, "statistics.txt"), sep="\t")
        ax = data.boxplot(column="SS")

        # Adjust font size
        plt.xticks(fontsize=8)

        # Rotate and align the x-axis labels vertically
        xticklabels = [
            label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(xticklabels)
        plt.savefig(os.path.join(outfolder, "boxplot.png"))


def polarplot(matches, outfile, SS=None):
    THETA = []
    R = []
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]

        geod = pyproj.Geod(ellps="WGS84")

        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(
            lon_obs[0], lat_obs[0], lon_obs[-1], lat_obs[-1]
        )
        vel0 = distance0 / 86400
        # simulation
        azimuth_betha, a2, distance1 = geod.inv(
            lon_mod[0], lat_mod[0], lon_mod[-1], lat_mod[-1]
        )
        vel1 = distance1 / 86400

        # calculate the coordinates in the polar plot
        theta = (azimuth_betha - azimuth_alpha + 360) % 360
        r = vel1 / vel0 if vel0 > 0 else 100
        THETA.append(theta)
        R.append(r)

    colors = [
        "#F8831C",  # orange
        "#1C26F8",  # blue
        "#000000",  # black
    ]

    R = np.array(R)
    THETA = np.array(THETA)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N", offset=0)
    ax.set_theta_direction(-1)
    ax.grid(True)
    # Set radial ticks
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1, 2]
    # Set radial tick labels, keep empty strings for values less than 1
    radial_ticklabels = ["" if tick < 1 else str(tick) for tick in radial_ticks]
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels(radial_ticklabels)
    circle1 = plt.Circle((0, 0), 0.2, color="r")
    ax.add_patch(circle1)
    if SS is None: #What does this condition mean?

        mask1 = np.where(np.logical_or(R < 0.5, np.logical_and(R > 1.5, R <= 2)))[0]
        mask2 = np.where(np.logical_and(R >= 0.5, R <= 1.5))[0]
        mask3 = np.where(R > 2)[0]
        # orange
        ax.scatter(
            THETA[mask1],
            R[mask1],
            zorder=10,
            color=colors[0],
            label="R < 0.5 or R in [1.5,2] ",
        )
        # blue
        ax.scatter(
            THETA[mask2], R[mask2], zorder=10, color=colors[1], label="R in [0.5,1.5]"
        )
        # black
        ax.scatter(
            THETA[mask3],
            R[mask3] / R[mask3] + 1,
            zorder=10,
            color=colors[2],
            label="R > 2",
        )

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.1))
    else:
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=1)  # Assuming SS values range from 0 to 1
        mask = np.where(R <= 2)[0]
        colors = cmap(norm(SS[mask]))
        ax.scatter(
            THETA[mask],
            R[mask],
            zorder=10,
            color=colors,
        )
        mask = np.where(R > 2)[0]
        colors = cmap(norm(SS[mask]))
        ax.scatter(
            THETA[mask],
            R[mask] / R[mask] + 1,
            zorder=10,
            color=colors,
        )
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label("SS Value")
    ax.set_rmin(0)
    ax.set_rmax(2.1)
    ax.set_title(f"{len(R)} points")
    plt.savefig(outfile, bbox_inches="tight")
    # plt.show()


def polar_to_cartesian(theta, r):
    """Convert polar coordinates to Cartesian coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polarplot2(matches, outfile, SS=None): #USE! THIS ONE
    THETA = []
    R = []
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]

        geod = pyproj.Geod(ellps="WGS84")

        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(
            lon_obs[0], lat_obs[0], lon_obs[-1], lat_obs[-1]
        )
        vel0 = distance0 / 86400
        # simulation
        azimuth_betha, a2, distance1 = geod.inv(
            lon_mod[0], lat_mod[0], lon_mod[-1], lat_mod[-1]
        )
        vel1 = distance1 / 86400

        # calculate the coordinates in the polar plot
        theta = (azimuth_betha - azimuth_alpha + 360) % 360
        r = vel1 / vel0 if vel0 > 0 else 10
        THETA.append(theta)
        R.append(r)

    R = np.array(R)
    r_avg_with_outsiders = np.mean(R)
    THETA = np.array(THETA)
    theta_avg_with_outsiders = circmean(THETA)
    print(theta_avg_with_outsiders, r_avg_with_outsiders, np.max(R))
    # print(len(SS),len(R))
    outsiders = (R > 3).sum() / len(R) * 100                 #why R<3, why 3?
    # Convert polar coordinates to Cartesian coordinates
    X, Y = polar_to_cartesian(np.pi / 2 - np.radians(THETA[R <= 3]), R[R <= 3])
    x_outsiders, y_outsiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_with_outsiders), r_avg_with_outsiders
    )
    r_avg_insiders = np.mean(R[R <= 3])
    theta_avg_insiders = circmean(THETA[R <= 3], high=360, low=0)
    x_insiders, y_insiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_insiders), r_avg_insiders
    )

    # Plot the density
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(X, Y, cmap="viridis", gridsize=50, bins="log")
    plt.colorbar(hb, label="Density")
    ax.scatter(0, 1, marker="x", s=200, c="r", label="target")
    ax.scatter(
        x_insiders,
        y_insiders,
        marker="o",
        s=50,
        c="r",
        label=f"mean without outsiders (R={np.round(r_avg_insiders,1)},theta={np.round(theta_avg_insiders,1)})",
    )
    ax.plot(
        np.linspace(-3, 3, 100),
        np.tan(np.pi / 2 - np.deg2rad(theta_avg_insiders)) * np.linspace(-3, 3, 100),
        color="k",
        linestyle="--",
        label=f"average deviation : {np.round(theta_avg_insiders,1)}deg",
    )
    ax.hlines(0, xmin=-5, xmax=5, linestyle=":", color="k", alpha=0.5)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    circle1 = plt.Circle((0, 0), 1, edgecolor="k", linestyle="--", fc=None, fill=False)
    ax.add_patch(circle1)
    plt.title(f"Polar Plot \n {np.round(outsiders)}% of points out of plot domain")
    plt.legend(loc="lower left")
    plt.gca().set_aspect(
        "equal", adjustable="box"
    )  # Set aspect ratio to make it look polar
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def polarplot_contour(matches, outfile, c=0.2, SS=None): #This should be the most up to date verison, USE!
    """Used to generate a polarplot from a list of matches

    Args:
        matches (_type_): _description_
        outfile (_type_): _description_
        c (float, optional): _description_. Defaults to 0.2.
        SS (_type_, optional): _description_. Defaults to None.
    """
    THETA = []
    R = []
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]

        geod = pyproj.Geod(ellps="WGS84")

        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(
            lon_obs[0], lat_obs[0], lon_obs[-1], lat_obs[-1]
        )
        vel0 = distance0 / 86400
        # simulation
        azimuth_betha, a2, distance1 = geod.inv(
            lon_mod[0], lat_mod[0], lon_mod[-1], lat_mod[-1]
        )
        vel1 = distance1 / 86400

        # calculate the coordinates in the polar plot
        theta = (azimuth_betha - azimuth_alpha + 360) % 360
        r = vel1 / vel0 if vel0 > 0 else 10
        THETA.append(theta)
        R.append(r)

    R = np.array(R)
    r_avg_with_outsiders = np.mean(R)
    THETA = np.array(THETA)
    theta_avg_with_outsiders = circmean(THETA)
    print(theta_avg_with_outsiders, r_avg_with_outsiders, np.max(R))
    # print(len(SS),len(R))
    outsiders = (R > 3).sum() / len(R) * 100
    # Convert polar coordinates to Cartesian coordinates
    x, y = polar_to_cartesian(np.pi / 2 - np.radians(THETA[R <= 3]), R[R <= 3])
    x_outsiders, y_outsiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_with_outsiders), r_avg_with_outsiders
    )
    r_avg_insiders = np.mean(R[R <= 3])
    theta_avg_insiders = circmean(THETA[R <= 3], high=360, low=0)
    x_insiders, y_insiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_insiders), r_avg_insiders
    )
    grid_size = 100
    percentages = [0.9, 0.75, 0.5, 0.25, 0.1, 0]
    points_insiders = np.vstack((x, y)).T
    X, Y, density_grid = create_density_grid(points_insiders, grid_size, c)
    levels = find_contour_levels(density_grid, percentages)
    # Plot the density
    fig, ax = plt.subplots(figsize=(6, 6))

    contourf = ax.contourf(
        X,
        Y,
        density_grid,
        levels=levels,
        cmap="viridis",
        alpha=0.9,
    )
    contour = ax.contour(
        X,
        Y,
        density_grid,
        levels=levels,
        colors="k",
        linewidths=(0.5,),
    )
    clabels = ax.clabel(
        contour,
        inline=True,
        fontsize=12,
        fmt={level: f"{int(p*100)}%" for level, p in zip(levels, percentages)},
    )
    ## Customize clabels
    for txt in clabels:
        txt.set_bbox(
            dict(facecolor="white", edgecolor=txt.get_color(), boxstyle="round,pad=0.2")
        )
        txt.set_color("black")

    ax.scatter(0, 1, marker="x", s=200, c="r", label="target", zorder=20)
    ax.scatter(
        x_insiders,
        y_insiders,
        marker="o",
        s=50,
        c="r",
        label=f"mean without ouliers (R={np.round(r_avg_insiders,1)},theta={np.round(theta_avg_insiders,1) if theta_avg_insiders<180 else np.round(theta_avg_insiders-360,1)}°)",
        zorder=30,
    )
    ax.scatter(
        x, y, alpha=0.5, marker="v", color="white", edgecolor="k", label="data points"
    )
    ax.plot(
        np.linspace(-3, 3, 100),
        np.tan(np.pi / 2 - np.deg2rad(theta_avg_insiders)) * np.linspace(-3, 3, 100),
        color="k",
        linestyle="--",
        label=f"average deviation : {np.round(theta_avg_insiders,1) if theta_avg_insiders<180 else np.round(theta_avg_insiders-360,1)}°",
    )
    ax.hlines(0, xmin=-5, xmax=5, linestyle=":", color="k", alpha=0.5)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    circle1 = plt.Circle(
        (0, 0), 1, edgecolor="r", linestyle="--", fc=None, fill=False, zorder=30
    )
    ax.add_patch(circle1)
    plt.legend(loc="lower left", fontsize=12)
    plt.gca().set_aspect(
        "equal", adjustable="box"
    )  # Set aspect ratio to make it look polar
    plt.savefig(outfile, bbox_inches="tight", dpi=400)
    plt.show()


def plot_current_map(data, outfolder, model):
    colors = [
        "#f7f7f7",
        "#542788",
        "#998ec3",
        "#d8daeb",
        "#fee0b6",
        "#f1a340",
        "#b35806",
    ]
    custom_cmap = ListedColormap(colors)

    data.loc[:, "LON"] = (data.loc[:, "LON"] + 360) % 360
    data = data.sort_values(by="LON")
    lon = data.loc[:, "LON"].values
    lat = data.loc[:, "LAT"].values
    SkillScore = data.loc[:, "SS"].values

    # plt.hist(lat,bins=100)
    # plt.show()

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.NorthPolarStereo()},
        figsize=(6, 6),
        dpi=200,
    )
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.right_labels = False

    # Define the number of chunks
    num_chunks = 100
    # Define radius
    radius = 0.5

    # Calculate the number of lon points in each chunk
    chunk_size = len(lon) // num_chunks
    # Iterate over chunks
    for i in range(num_chunks):
        # Calculate the start and end indices for the chunk
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(lon)

        # Extract lon and lat values for the chunk
        lon_chunk = lon[start_idx:end_idx]
        lmin = lon_chunk.min()
        lmax = lon_chunk.max()
        # if lmin < l1 - 20 or lmax > l2 + 20:
        #     continue
        lat_chunk = lat[start_idx:end_idx]
        ss_chunk = SkillScore[start_idx:end_idx]

        # Calcul des limites
        lon_min = max(0, lon_chunk.min() - 1)
        lon_max = min(360, lon_chunk.max() + 1)
        lat_min = max(-90, lat_chunk.min() - 1)
        lat_max = min(90, lat_chunk.max() + 1)
        grid_resolution = 0.1

        # Create lon-lat grid for the chunk
        lon_grid, lat_grid = np.meshgrid(
            np.arange(lon_min, lon_max, grid_resolution),
            np.arange(lat_min, lat_max, grid_resolution),
        )
        # Calcul des distances à tous les points de données pour le chunk
        distances_chunk = np.sqrt(
            (lon_chunk[:, None, None] - lon_grid) ** 2
            + (lat_chunk[:, None, None] - lat_grid) ** 2
        )
        # Initialize third grid
        third_grid = np.full(lon_grid.shape, np.nan)

        # Vérifier si un point de données est dans le rayon pour chaque point de la grille
        within_radius = distances_chunk <= radius
        mask_isData = np.any(within_radius, axis=0)
        # print(np.where(mask_isData))

        average_SS = np.nansum(
            np.reshape(ss_chunk, (len(ss_chunk), 1, 1)) * within_radius, axis=0
        ) / np.nansum(within_radius, axis=0)
        # print(np.nanmin(average_SS), np.nanmax(average_SS), np.nanmean(average_SS))

        thresholds = [0, 0.005, 0.1, 0.2, 0.3, 0.5, 1]

        # Assigner les valeurs en fonction des seuils
        for i, (threshold1, threshold2) in enumerate(
            zip(thresholds[:-1], thresholds[1:])
        ):
            if i < 3:
                inThreshold = np.logical_and(
                    average_SS < threshold2, threshold1 < average_SS
                )
                third_grid[mask_isData & inThreshold] = i + 1
            else:
                inThreshold = np.logical_and(
                    average_SS < threshold2, threshold1 < average_SS
                )
                third_grid[mask_isData & inThreshold] = i + 1

        # mp = plt.pcolormesh(
        #     lon_grid,
        #     lat_grid,
        #     third_grid,
        #     cmap=custom_cmap,
        #     transform=ccrs.PlateCarree(),
        #     vmin=0,
        #     vmax=6,
        # )

        contour = plt.contourf(
            lon_grid,
            lat_grid,
            average_SS,
            # cmap=custom_cmap,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            vmin=0,
            vmax=0.4,
        )

    cbar = fig.colorbar(
        contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
    )

    ax.set_extent([0, 360, 60, 90], ccrs.PlateCarree())
    plt.savefig(
        os.path.join(outfolder, f"{model}_SS_map_global.png"),
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.show()


def plot_ensemble_mean(ensemble_file, projection=None, **kwargs): #interessting, USE!
    # Open the NetCDF file using netCDF4.Dataset
    with Dataset(ensemble_file, mode="r") as data:
        # Extract longitude and latitude
        lon = data["lon"][:]
        lat = data["lat"][:]

        # Extract coefficients and dimensions
        Ca = data["wind_drag_coeff"][:]
        Co = data["water_drag_coeff"][:]
        Lib = data["length"][:]
        Wib = data["width"][:]
        Sailib = data["sail"][:]
        Draftib = data["draft"][:]

        # Compute f
        rho_air = 1.2
        rho_water = 1000
        k = rho_air * Ca * Sailib * Lib / (rho_water * Co * Draftib * Lib)
        f = np.sqrt(k) / (1 + np.sqrt(k))

        # Compute mean longitude and latitude
        lon_mean = circmean(lon, high=360, low=0, axis=0)
        lat_mean = np.mean(lat, axis=0)

    # Create the plot
    if projection is None:
        projection = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 10))

    # Add coastlines and land features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # Customize gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    # Normalize f values to the range [0, 1] for the colormap
    norm = Normalize(vmin=np.min(f), vmax=np.max(f))
    cmap = plt.get_cmap("viridis")

    # Plot each track with colors according to the value of f
    for i in range(len(lon)):
        color = cmap(norm(f[i, 0]))
        ax.plot(lon[i], lat[i], transform=ccrs.PlateCarree(), color=color, alpha=0.4)

    # Plot the mean track in blue dashed line
    ax.plot(
        lon_mean,
        lat_mean,
        transform=ccrs.PlateCarree(),
        color="blue",
        linestyle="--",
        label="Ensemble mean",
    )
    ax.scatter(
        lon_mean[-1],
        lat_mean[-1],
        color="b",
        label="average end position",
        transform=ccrs.PlateCarree(),
        zorder=9,
    )
    ax.scatter(
        lon_mean[0],
        lat_mean[0],
        color="b",
        marker="v",
        label="average initial position",
        transform=ccrs.PlateCarree(),
        zorder=9,
    )

    # Add the legend for the mean track
    ax.legend(loc="lower left")

    # Add a colorbar for the f values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="f value")

    # Check for extent parameters in kwargs and set the extent if provided
    if "extent" in kwargs:
        ax.set_extent(kwargs["extent"], crs=ccrs.PlateCarree())

    # Add a title
    plt.title("Ensemble Mean Track with f Values", fontsize=16)

    # Show the plot
    plt.show()


def gaussian_kernel(r, sigma):
    return np.exp(-(r**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def calculate_average_distance(points):
    if len(points) > 1000:
        distances = pdist(points[::50])
    else:
        distances = pdist(points)
    d_avg = np.mean(distances)
    return d_avg


def find_contour_levels(density_grid, percentages):
    sorted_density = np.sort(density_grid.ravel())[::-1]
    cumsum_density = np.cumsum(sorted_density)
    total_density = cumsum_density[-1]
    levels = [
        sorted_density[np.searchsorted(cumsum_density, p * total_density)]
        for p in percentages
    ]
    levels.sort()  # S'assurer que les niveaux sont dans l'ordre croissant
    return levels


def create_density_grid(points, grid_size=100, c=0.5):
    points = points[~np.isnan(points).any(axis=1)]
    # Calculer la distance moyenne entre les points
    d_avg = calculate_average_distance(points) + 1
    # Déterminer sigma en fonction de d_avg et du paramètre c
    sigma = c * d_avg

    # Déterminer les limites de la grille
    x_min, x_max = min(points[:, 0]) - 2 * sigma, max(points[:, 0]) + 2 * sigma
    y_min, y_max = min(points[:, 1]) - 2 * sigma, max(points[:, 1]) + 2 * sigma

    # Créer une grille de coordonnées
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)

    # Initialiser la grille de densité
    density_grid = np.zeros((grid_size, grid_size))

    # Nombre de points
    N = len(points)

    # Appliquer la distribution gaussienne à chaque point
    for point in points:
        dx = X - point[0]
        dy = Y - point[1]
        r = np.sqrt(dx**2 + dy**2)
        density_grid += gaussian_kernel(r, sigma) / N

    return X, Y, density_grid


def plot_contour_ensemble(
    ensemble_file,
    observation=None,
    grid_size=100,
    c=0.5,
    percentages=[0.9, 0.75, 0.5, 0.25],
    projection=None,
    membres=False,
    ax=None,
    **kwargs,
):
    """Function used to plot the ensemble contour plot

    Args:
        ensemble_file (_type_): _description_
        observation (_type_, optional): _description_. Defaults to None.
        grid_size (int, optional): _description_. Defaults to 100.
        c (float, optional): _description_. Defaults to 0.5.
        percentages (list, optional): _description_. Defaults to [0.9, 0.75, 0.5, 0.25].
        projection (_type_, optional): _description_. Defaults to None.
        membres (bool, optional): _description_. Defaults to False.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Open the NetCDF file using netCDF4.Dataset
    with Dataset(ensemble_file, mode="r") as data:
        # Extract longitude and latitude
        lon = data["lon"][:, :]
        lat = data["lat"][:, :]

        # Compute mean longitude and latitude
        lon_max = np.nanmax(lon[:, -1])
        lon_min = np.nanmin(lon[:, -1])
        lat_max = np.nanmax(lat[:, -1])
        lat_min = np.nanmin(lat[:, -1])
        lon_mean = circmean(lon, high=360, low=0, axis=0, nan_policy="omit")
        lat_mean = np.nanmean(lat, axis=0)
        lon_start_mean = circmean(
            data["lon"][:, 0], high=360, low=0, axis=0, nan_policy="omit"
        )
        lat_start_mean = np.nanmean(data["lat"][:, 0], axis=0)
        lon_end_mean = circmean(lon[:, -1], high=360, low=0, axis=0, nan_policy="omit")
        lat_end_mean = np.nanmean(lat[:, -1], axis=0)

    # Convert lon/lat to Cartesian coordinates
    transformer = Transformer.from_proj(
        "epsg:4326", "epsg:32661", always_xy=True
    )  # North Polar Stereographic
    x_start, y_start = transformer.transform(lon[:, 0], lat[:, 0])
    x_end, y_end = transformer.transform(lon[:, -1], lat[:, -1])

    # Combine x and y into a single array for density calculation
    points_start = np.vstack((x_start, y_start)).T
    points_end = np.vstack((x_end, y_end)).T

    # Create the density grid
    X_start, Y_start, density_grid_start = create_density_grid(
        points_start, grid_size, c
    )
    X_end, Y_end, density_grid_end = create_density_grid(points_end, grid_size, c)

    # Calculate contour levels for the given percentages
    levels_start = find_contour_levels(density_grid_start, percentages + [0])
    levels_end = find_contour_levels(density_grid_end, percentages + [0])

    # Create the plot with Cartopy
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 8))
    else:
        pass

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    # Transform X, Y back to lon/lat for plotting
    inv_transformer = Transformer.from_proj("epsg:32661", "epsg:4326", always_xy=True)
    lon_grid_start, lat_grid_start = inv_transformer.transform(X_start, Y_start)
    lon_grid_end, lat_grid_end = inv_transformer.transform(X_end, Y_end)

    # Plot the density grid contours
    if levels_start[-1] > levels_start[0]:
        contourf = ax.contourf(
            lon_grid_start,
            lat_grid_start,
            density_grid_start,
            levels=levels_start,
            cmap="plasma_r",
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )
        contour = ax.contour(
            lon_grid_start,
            lat_grid_start,
            density_grid_start,
            levels=levels_start,
            colors="k",
            transform=ccrs.PlateCarree(),
        )

    contourf = ax.contourf(
        lon_grid_end,
        lat_grid_end,
        density_grid_end,
        levels=levels_end,
        cmap="plasma_r",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
    )
    contour = ax.contour(
        lon_grid_end,
        lat_grid_end,
        density_grid_end,
        levels=levels_end,
        colors="k",
        transform=ccrs.PlateCarree(),
    )

    # Scatter plot of the members
    if membres:
        ax.scatter(
            lon[::, -1],
            lat[::, -1],
            s=1,
            color="k",
            marker="o",
            alpha=0.07,
            transform=ccrs.PlateCarree(),
            zorder=0,
        )
    # Plot the mean track in blue dashed line
    ax.plot(
        lon_mean,
        lat_mean,
        transform=ccrs.PlateCarree(),
        color="b",
        linestyle="--",
        label="Ensemble mean drift",
    )

    # Plot the mean end position
    ax.scatter(
        lon_end_mean,
        lat_end_mean,
        color="b",
        label="average end position",
        transform=ccrs.PlateCarree(),
        zorder=10,
        edgecolor="white",
    )
    ax.scatter(
        lon_start_mean,
        lat_start_mean,
        color="b",
        marker="v",
        label="average initial position",
        transform=ccrs.PlateCarree(),
        zorder=9,
    )
    if not (observation is None):
        lon_obs, lat_obs = observation
        ax.plot(
            lon_obs,
            lat_obs,
            transform=ccrs.PlateCarree(),
            color="orange",
            linestyle="-",
            label="Observation",
        )
        ax.scatter(
            lon_obs[0],
            lat_obs[0],
            transform=ccrs.PlateCarree(),
            color="orange",
            marker="v",
        )
        ax.scatter(
            lon_obs[-1],
            lat_obs[-1],
            transform=ccrs.PlateCarree(),
            color="orange",
            marker="o",
            zorder=10,
            edgecolor="white",
        )
        # Convert observation lon/lat to Cartesian coordinates
        x_obs, y_obs = transformer.transform(observation[0][-1], observation[1][-1])

        # Calculate density at observation point
        arg_x = np.argmin(np.abs(X_end - x_obs), axis=1)
        arg_y = np.argmin(np.abs(Y_end - y_obs), axis=0)
        obs_density_end = density_grid_end[arg_y[0]][arg_x[0]]

        # Check which contour level encapsulates the observation
        obs_level_end = 0
        for i, level in enumerate(levels_end):
            if obs_density_end >= level:
                obs_level_end = percentages[i]

        print(
            f"Observation is encapsulated by contour level :  {int(obs_level_end*100)}%"
        )
    # lon_min = lon_min - 0.1 * (lon_mean.max() - lon_mean.min())
    # lon_max = lon_max + 0.1 * (lon_mean.max() - lon_mean.min())
    # lat_min = lat_min - 0.1 * (lat_mean.max() - lat_mean.min())
    # lat_max = lat_max + 0.1 * (lat_mean.max() - lat_mean.min())
    lon_min = lon_mean.min() - 0.5 * (lon_mean.max() - lon_mean.min())
    lon_max = lon_mean.max() + 0.5 * (lon_mean.max() - lon_mean.min())
    lat_min = lat_mean.min() - 0.5 * (lat_mean.max() - lat_mean.min())
    lat_max = lat_mean.max() + 0.5 * (lat_mean.max() - lat_mean.min())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.set_box_aspect(1)

    return ax, contourf, contour


def compute_ensemble_dispersion(x, y, x0, obs):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        x0 (_type_): _description_
        obs (_type_): _description_

    Returns:
        _type_: mu_b,mu_r,R,err
    """
    r = np.sqrt((x - x0[0]) ** 2 + (y - x0[1]) ** 2)
    B = np.array([np.mean(x), np.mean(y)])  # ensemble mean
    b = np.sqrt((x - B[0]) ** 2 + (y - B[1]) ** 2)

    mu_r = np.mean(r)  # average distance from the initial position
    mu_b = np.mean(b)  # dispersion of the ensemble
    sig_b1 = np.std((x - B[0]))  # dispersion en x
    sig_b2 = np.std((y - B[1]))  # dispersion en y
    R = sig_b2 / sig_b1  # anisotropy
    err = B - obs  # vector error = position of the observation in B coordinates

    return (mu_b, mu_r, R, err)


def determine_contour_level(
    ensemble_file,
    observation,
    ensemble: tuple = None,
    grid_size=100,
    c=0.5,
    percentages=[0.9, 0.75, 0.5, 0.25],
):
    if ensemble is None:
        # Open the NetCDF file using netCDF4.Dataset
        with Dataset(ensemble_file, mode="r") as data:
            # Extract longitude and latitude
            lon = data["lon"][:, :]
            lat = data["lat"][:, :]
    else:
        lon, lat = ensemble

    # Convert lon/lat to Cartesian coordinates
    transformer = Transformer.from_proj(
        "epsg:4326", "epsg:32661", always_xy=True
    )  # North Polar Stereographic
    x_end, y_end = transformer.transform(lon[:, -1], lat[:, -1])

    # Combine x and y into a single array for density calculation
    points_end = np.vstack((x_end, y_end)).T

    # Create the density grid
    X_end, Y_end, density_grid_end = create_density_grid(points_end, grid_size, c)

    # Calculate contour levels for the given percentages
    levels_end = find_contour_levels(density_grid_end, percentages + [0])

    # Convert observation lon/lat to Cartesian coordinates
    x_obs0, y_obs0 = transformer.transform(observation[0][0], observation[1][0])
    x0 = np.array([x_obs0, y_obs0])
    x_obs, y_obs = transformer.transform(observation[0][-1], observation[1][-1])

    # Calculate density at observation point
    arg_x = np.argmin(np.abs(X_end - x_obs), axis=1)
    arg_y = np.argmin(np.abs(Y_end - y_obs), axis=0)
    obs_density_end = density_grid_end[arg_y[0]][arg_x[0]]

    # Check which contour level encapsulates the observation
    obs_level_end = 0
    for i, level in enumerate(levels_end):
        if obs_density_end >= level:
            obs_level_end = percentages[i]

    return int(obs_level_end * 100), compute_ensemble_dispersion(
        x_end, y_end, x0, np.array([x_obs, y_obs])
    )
