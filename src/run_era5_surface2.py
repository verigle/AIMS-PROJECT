import xarray as xr
import gcsfs
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses

# Data
fs = gcsfs.GCSFileSystem(token="anon")
store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)

# Select time range
start_time, end_time = '2022-10-01', '2023-01-31'

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

# Surface variable names and plot titles
variables = ["2t", "10u", "10v", "msl"]
plots_titles = [
    "Two-meter temperature two steps forward prediction: RMSES",
    "Ten-meter eastward wind speed two steps forward prediction: RMSES",
    "Ten-meter southward wind speed two steps forward prediction: RMSES",
    "Mean sea-level pressure two steps forward prediction: RMSES"
]

selected_times = sliced_era5_world.time
world_latitudes, world_longitudes = sliced_era5_world.latitude, sliced_era5_world.longitude
sa_latitudes, sa_longitudes = sliced_era5_sa.latitude, sliced_era5_sa.longitude

# Compute RMSE weights
world_rmse_weights = rmse_weights(world_latitudes, world_longitudes)[1:, :]
sa_rmse_weights = rmse_weights(sa_latitudes, sa_longitudes)

print("Data processing done")

# Initialize lists for storing results
pred_dates_list = []
world_rmses_list = {var: [] for var in variables}
sa_rmses_list = {var: [] for var in variables}

# Main loop to calculate RMSE for each variable
for i in range(len(sliced_era5_world.time) - 3):
    # Get current and previous time step data
    time_slice = slice(selected_times[i], selected_times[i + 1])
    world_feature_data = sliced_era5_world.sel(time=time_slice)
    sa_feature_data = sliced_era5_sa.sel(time=time_slice)
    
    # Get next time step data
    target_time_slice = slice(selected_times[i + 2], selected_times[i + 3])
    world_target_data = sliced_era5_world.sel(time=target_time_slice)
    sa_target_data = sliced_era5_sa.sel(time=target_time_slice)

    # Get feature and target data for both regions
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

    # Make predictions for both regions
    world_predictions = predict_fn(batch=world_feature_batch)
    sa_predictions = predict_fn(batch=sa_feature_batch)

    # Compute RMSE for each variable
    for var in variables:
        world_rmses, world_pred_dates = rmse_fn(
            predictions=world_predictions, target_batch=world_target_batch,
            var_name=var, weigths=world_rmse_weights
        )
        sa_rmses, _ = rmse_fn(
            predictions=sa_predictions, target_batch=sa_target_batch,
            var_name=var, weigths=sa_rmse_weights, area="sa"
        )

        world_rmses_list[var].append(world_rmses)
        sa_rmses_list[var].append(sa_rmses)

    pred_dates_list.append(world_pred_dates)

    # Print progress every 10 iterations
    if (i + 1) % 10 == 0:
        print(f"Iteration {i + 1} completed")

# Plot RMSE for each variable
for idx, var in enumerate(variables):
    plot_rmses(
        var, world_rmses_list[var], sa_rmses_list[var],
        figsize=(12, 8), fontsize=18, date_ranges=pred_dates_list,
        title=plots_titles[idx], save_path="../report/era5"
    )

print("Plotting done")
