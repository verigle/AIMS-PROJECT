import os
import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, create_hrest0_batch

import xarray as xr
import gcsfs
import numpy as np
from aurora import AuroraSmall, Batch, Metadata, rollout

import torch
import numpy as np
# Data Loading
fs = gcsfs.GCSFileSystem(token="anon")

# Load the datasets from Google Cloud Storage
store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)

store_era5 = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store_era5, consolidated=True, chunks=None)

# Time Range Selection
# Select time range
start_time = '2022-09-01'
end_time = '2022-12-31'
# end_time = '2022-11-03'
# USA and South Africa data slicing - ADJUSTED SIZE FOR USA
# Use a larger area for USA to ensure enough points for convolution
sliced_hrest0_usa = full_hrest0.sel(time=slice(start_time, end_time),
                                   latitude=slice(20.00, 54.99),  # Expanded USA latitude range
                                   longitude=slice(230, 299.99))  # Expanded USA longitude range

# Keep South Africa as is since it was working before
sliced_hrest0_sa = full_hrest0.sel(time=slice(start_time, end_time), 
                                  latitude=slice(-37.75, -22.00), 
                                  longitude=slice(15.25, 35.00))

# Make corresponding adjustments for ERA5 data
sliced_era5_usa = full_era5.sel(time=slice(start_time, end_time),
                               latitude=slice(54.99, 20.00),  # Note the flip in order for ERA5
                               longitude=slice(230, 299.99))

sliced_era5_sa = full_era5.sel(time=slice(start_time, end_time),
                              latitude=slice(-22.00, -37.75),
                              longitude=slice(15.25, 35.00))

# Constants
STATIC_VARIABLES = ["land_sea_mask", "soil_type", "geopotential_at_surface"]

selected_times = sliced_hrest0_usa.time

# Compute RMSE weights for USA and South Africa
usa_rmse_weights = rmse_weights(sliced_hrest0_usa.latitude, sliced_hrest0_usa.longitude)
sa_rmse_weights = rmse_weights(sliced_era5_sa.latitude, sliced_hrest0_sa.longitude)

# Atmospheric levels and variables to evaluate
atmos_levels_idx = [6, 12]
atmos_level_names = ["400hPa", "1000hPa"]
atmos_vars_names = ["t", "u", "v", "q", "z"]
plots_titles = [
    "Temperature in K two steps forward prediction: RMSES",
    "Eastward wind speed two steps forward prediction: RMSES",
    "Southward wind speed two steps forward prediction: RMSES",
    "Specific humidity two steps forward prediction: RMSES",
    "Geopotential two steps forward prediction: RMSES"
]

model_initial = AuroraSmall(
    use_lora=False,
    autocast=True,
    stabilise_level_agg=True
)
model_initial.load_state_dict(torch.load('../model/aurora-0.25-small-pretrained.pth'))

################### Main Loop ###################
for atmos_level_idx, atmos_level_name in zip(atmos_levels_idx, atmos_level_names):
    # Initialize lists to store RMSE results for all variables at the current atmospheric level
    usa_rmses_list = {var: [] for var in atmos_vars_names}
    sa_rmses_list = {var: [] for var in atmos_vars_names}
    pred_dates_list = []

    for i in range(len(selected_times) - 3):
        # Extract feature and target data for both USA and South Africa regions
        usa_feature_hrest0_data = sliced_hrest0_usa.sel(time=slice(selected_times[i], selected_times[i+1]))
        usa_target_hrest0_data = sliced_hrest0_usa.sel(time=slice(selected_times[i+2], selected_times[i+3]))
        sa_feature_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
        sa_target_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))

        usa_feature_era5_data = sliced_era5_usa.sel(time=slice(selected_times[i], selected_times[i+1]))
        usa_target_era5_data = sliced_era5_usa.sel(time=slice(selected_times[i+2], selected_times[i+3]))
        sa_feature_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
        sa_target_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))
        
        # Extract surface, atmospheric, and static features and target data
        usa_feature_surface_data, usa_target_surface_data = get_surface_feature_target_data(usa_feature_hrest0_data, usa_target_hrest0_data)
        usa_feature_atmos_data, usa_target_atmos_data = get_atmos_feature_target_data(usa_feature_hrest0_data, usa_target_hrest0_data)
        usa_feature_static_data, usa_target_static_data = get_static_feature_target_data(usa_feature_era5_data, usa_target_era5_data, STATIC_VARIABLES)
        
        sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
        sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
        sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data, STATIC_VARIABLES)
        
        # Create batches for both regions
        usa_feature_batch = create_hrest0_batch(usa_feature_surface_data, usa_feature_atmos_data, usa_feature_static_data)
        usa_target_batch = create_hrest0_batch(usa_target_surface_data, usa_target_atmos_data, usa_target_static_data)
        sa_feature_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
        sa_target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)
        
        # Generate predictions for both regions
        sa_predictions = predict_fn(model=model_initial, batch=sa_feature_batch)
        usa_predictions = predict_fn(model=model_initial, batch=usa_feature_batch)
        
        # Compute RMSE for each variable
        for var, title in zip(atmos_vars_names, plots_titles):
            usa_rmses, usa_pred_dates = rmse_fn(predictions=usa_predictions, 
                                              target_batch=usa_target_batch, 
                                              var_name=var, weigths=usa_rmse_weights,
                                              var_type="atmosphere")
            sa_rmses, sa_pred_dates = rmse_fn(predictions=sa_predictions,
                                            target_batch=sa_target_batch,
                                            var_name=var, weigths=sa_rmse_weights,
                                            var_type="atmosphere", area="sa")
            
            # Append RMSE for the current variable and current iteration `i`
            usa_rmses_list[var].append(usa_rmses)
            sa_rmses_list[var].append(sa_rmses)
        
        # Store the prediction dates for plotting
        pred_dates_list.append(usa_pred_dates)
        
        if (i+1) % 10 == 0:
            print(f"Iterations {i+1} for atmospheric level {atmos_level_name}")
    
    # After processing all iterations for this atmospheric level, plot RMSE results for all variables
    for var, title in zip(atmos_vars_names, plots_titles):
        plot_rmses(var, usa_rmses_list[var], sa_rmses_list[var], 
                   figsize=(12, 8), fontsize=18,
                   date_ranges=pred_dates_list, 
                   title=title,
                   save_path="../report/hrest0_usa",  # Changed save path to usa
                   atmos_level=atmos_level_name,
                   place="USA")  # Changed place label to USA
        print(f"Plot for {var} at {atmos_level_name} Done")

    print(f"Completed RMSE computation and plotting for atmospheric level {atmos_level_name}")