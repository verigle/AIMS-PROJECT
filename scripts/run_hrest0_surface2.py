import os
import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, create_hrest0_batch
import pandas as pd
import xarray as xr
import gcsfs

# Data
fs = gcsfs.GCSFileSystem(token="anon")

# Load datasets
store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)

store_era5 = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store_era5, consolidated=True, chunks=None)

# Select time range
start_time = '2022-11-03'
end_time = '2022-11-04'

# World and South Africa data slicing
sliced_hrest0_world = full_hrest0.sel(time=slice(start_time, end_time))
sliced_hrest0_sa = full_hrest0.sel(time=slice(start_time, end_time), 
                                   latitude=slice(-37.75, -22.00), 
                                   longitude=slice(15.25, 35.00))

sliced_era5_world = full_era5.sel(time=slice(start_time, end_time))
sliced_era5_sa = full_era5.sel(time=slice(start_time, end_time),
                               latitude=slice(-22.00, -37.75),
                               longitude=slice(15.25, 35.00))

# Constants
STATIC_VARIABLES = ["land_sea_mask", "soil_type", "geopotential_at_surface"]
surface_vars_names = ["2t", "10u", "10v", "msl"]
plots_titles = [
    "Two-meter temperature two steps forward prediction: RMSES",
    "Ten-meter eastward wind speed two steps forward prediction: RMSES",
    "Ten-meter southward wind speed two steps forward prediction: RMSES",
    "Mean sea-level pressure two steps forward prediction: RMSES"
]

selected_times = sliced_hrest0_world.time
# print(selected_times)

# Compute RMSE weights once
world_rmse_weights = rmse_weights(sliced_hrest0_world.latitude, sliced_hrest0_world.longitude)[1:, :]
sa_rmse_weights = rmse_weights(sliced_era5_sa.latitude, sliced_hrest0_sa.longitude)

# Initialize result lists for this iteration
world_rmses_list = {var: [] for var in surface_vars_names}
sa_rmses_list = {var: [] for var in surface_vars_names}
pred_dates_list = []
################### Main Loop ###################

for i in range(len(selected_times) - 3):
    
    
    # Get feature and target data for this timestep
    world_feature_hrest0_data = sliced_hrest0_world.sel(time=slice(selected_times[i], selected_times[i+1]))
    world_target_hrest0_data = sliced_hrest0_world.sel(time=slice(selected_times[i+2], selected_times[i+3]))
    sa_feature_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
    sa_target_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))

    world_feature_era5_data = sliced_era5_world.sel(time=slice(selected_times[i], selected_times[i+1]))
    world_target_era5_data = sliced_era5_world.sel(time=slice(selected_times[i+2], selected_times[i+3]))
    sa_feature_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
    sa_target_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))
    
    # Extract features and targets for all surface variables at once
    world_feature_surface_data, world_target_surface_data = get_surface_feature_target_data(world_feature_hrest0_data, world_target_hrest0_data)
    world_feature_atmos_data, world_target_atmos_data = get_atmos_feature_target_data(world_feature_hrest0_data, world_target_hrest0_data)
    world_feature_static_data, world_target_static_data = get_static_feature_target_data(world_feature_era5_data, world_target_era5_data, STATIC_VARIABLES)
    
    sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
    sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
    sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data, STATIC_VARIABLES)
    
    # Create batches for all surface variables at once
    world_feature_batch = create_hrest0_batch(world_feature_surface_data, world_feature_atmos_data, world_feature_static_data)
    world_target_batch = create_hrest0_batch(world_target_surface_data, world_target_atmos_data, world_target_static_data)
    sa_feature_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
    sa_target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)
    # Predictions for all surface variables
    world_predictions = predict_fn(batch=world_feature_batch)
    sa_predictions = predict_fn(batch=sa_feature_batch)
    print(f"prediction: {world_predictions}")
    
    # Compute RMSE for all surface variables at once
    # for var in surface_vars_names:
    #     world_rmses, world_pred_dates = rmse_fn(
    #         predictions=world_predictions, 
    #         target_batch=world_target_batch, 
    #         var_name=var, 
    #         weigths=world_rmse_weights
    #     )
    #     print( world_pred_dates)
    #     sa_rmses, sa_pred_dates = rmse_fn(
    #         predictions=sa_predictions, 
    #         target_batch=sa_target_batch, 
    #         var_name=var, 
    #         weigths=sa_rmse_weights, 
    #         area="sa"
    #     )
        
    #     # Append the results for this variable
    #     world_rmses_list[var].append(world_rmses)
    #     sa_rmses_list[var].append(sa_rmses)
    #     pred_dates_list[var].append(world_pred_dates)
        
    for var in surface_vars_names:
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
    
    if (i+1) % 10 == 0:
        print(f"Iterations {i+1}")
# Saving for checking

dict = {"dates": pred_dates_list, "rmses":world_rmses_list["2t"]}
df_rmses = pd.DataFrame(dict)
df_rmses.to_csv("rmses.csv")

# Plot results for each surface variable after all iterations
for var, title in zip(surface_vars_names, plots_titles):
    plot_rmses(
        var, 
        world_rmses_list[var], 
        sa_rmses_list[var], 
        figsize=(12, 8), 
        fontsize=18, 
        date_ranges=pred_dates_list, 
        title=title, 
        save_path="../report/hrest0", 
        atmos_level=None
    )
    print(f"Plot for {var} completed")

print("All plots done")
