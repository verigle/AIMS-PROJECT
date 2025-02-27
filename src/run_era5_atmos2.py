from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses

import xarray as xr
import gcsfs
import os

# Data
fs = gcsfs.GCSFileSystem(token="anon")
store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)

# Select time range
start_time, end_time = '2022-12-01', '2023-01-31'

# World Data
sliced_era5_world = full_era5.sel(time=slice(start_time, end_time))

# South Africa Data
lat_max, lat_min = -22.00, -37.75
lon_min, lon_max = 15.25, 35.00
sliced_era5_sa = full_era5.sel(
    time=slice(start_time, end_time),
    latitude=slice(lat_max, lat_min),
    longitude=slice(lon_min, lon_max)
)

# Atmospheric variables and levels
atmos_levels_idx = [0, 6, 12]
atmos_level_names = ["50hPa", "400hPa", "1000hPa"]
atmos_vars_names = ["t", "u", "v", "q", "z"]
plots_titles = [
    "Temperature in K two steps forward prediction: RMSES",
    "Eastward wind speed two steps forward prediction: RMSES",
    "Southward wind speed two steps forward prediction: RMSES",
    "Specific humidity two steps forward prediction: RMSES",
    "Geopotential two steps forward prediction: RMSES"
]

# Select times and lat/lon for weight calculation
selected_times = sliced_era5_world.time
world_latitudes, world_longitudes = sliced_era5_world.latitude, sliced_era5_world.longitude
sa_latitudes, sa_longitudes = sliced_era5_sa.latitude, sliced_era5_sa.longitude

# Compute RMSE weights
world_rmse_weights = rmse_weights(world_latitudes, world_longitudes)[1:, :]
sa_rmse_weights = rmse_weights(sa_latitudes, sa_longitudes)

# Main loop to compute RMSE for each atmospheric level
for atmos_level_idx, atmos_level_name in zip(atmos_levels_idx, atmos_level_names):
    # Initialize results container for this level
    world_rmses_list = {var: [] for var in atmos_vars_names}
    sa_rmses_list = {var: [] for var in atmos_vars_names}
    pred_dates_list = {var: [] for var in atmos_vars_names}
    
    iterations = 0
    for i in range(0, len(sliced_era5_world.time) - 3):
        iterations += 1

        # Get current and next time step data
        time_slice = slice(selected_times[i], selected_times[i + 1])
        world_feature_data = sliced_era5_world.sel(time=time_slice)
        sa_feature_data = sliced_era5_sa.sel(time=time_slice)

        target_time_slice = slice(selected_times[i + 2], selected_times[i + 3])
        world_target_data = sliced_era5_world.sel(time=target_time_slice)
        sa_target_data = sliced_era5_sa.sel(time=target_time_slice)

        # Get each type of data (surface, static atmosphere)
        world_feature_surface_data, world_target_surface_data = get_surface_feature_target_data(world_feature_data, world_target_data)
        world_feature_atmos_data, world_target_atmos_data = get_atmos_feature_target_data(world_feature_data, world_target_data)
        world_feature_static_data, world_target_static_data = get_static_feature_target_data(world_feature_data, world_target_data)

        sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_data, sa_target_data)
        sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_data, sa_target_data)
        sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_data, sa_target_data)

        # Create batches for both regions
        world_feature_batch = create_batch(world_feature_surface_data, world_feature_atmos_data, world_feature_static_data)
        world_target_batch = create_batch(world_target_surface_data, world_target_atmos_data, world_target_static_data)
        sa_feature_batch = create_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
        sa_target_batch = create_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)

        # Get predictions for both regions (for all variables at once)
        world_predictions = predict_fn(batch=world_feature_batch)
        sa_predictions = predict_fn(batch=sa_feature_batch)

        # Compute RMSE for all variables at once
        for var, title in zip(atmos_vars_names, plots_titles):
            world_rmses, world_pred_dates = rmse_fn(
                predictions=world_predictions, 
                target_batch=world_target_batch, 
                var_name=var, 
                atmos_level_idx=atmos_level_idx,
                weigths=world_rmse_weights, 
                var_type="atmosphere"
            )
            sa_rmses, sa_pred_dates = rmse_fn(
                predictions=sa_predictions, 
                target_batch=sa_target_batch, 
                var_name=var, 
                atmos_level_idx=atmos_level_idx,
                weigths=sa_rmse_weights, 
                var_type="atmosphere", 
                area="sa"
            )

            # Append results for the current variable and atmospheric level
            world_rmses_list[var].append(world_rmses)
            sa_rmses_list[var].append(sa_rmses)
            pred_dates_list[var].append(world_pred_dates)

        # Print progress
        if iterations % 10 == 0:
            print(f"Iterations {iterations} for atmospheric level {atmos_level_name}")

    # After all iterations for this atmospheric level, generate the plots for each variable
    for var, title in zip(atmos_vars_names, plots_titles):
        plot_rmses(
            var, 
            world_rmses_list[var], 
            sa_rmses_list[var], 
            figsize=(12, 8), 
            fontsize=18, 
            date_ranges=pred_dates_list[var], 
            title=title, 
            save_path="../report/era5", 
            atmos_level=atmos_level_name
        )

        print(f"Plot for {var} at {atmos_level_name} completed")

print("All plots done")
