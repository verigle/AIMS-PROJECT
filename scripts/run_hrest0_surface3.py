import os
import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, create_hrest0_batch
import pandas as pd
import xarray as xr
import gcsfs
from aurora import AuroraSmall, Batch, Metadata, rollout
import torch
import numpy as np

# Load datasets
fs = gcsfs.GCSFileSystem(token="anon")
store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)

store_era5 = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store_era5, consolidated=True, chunks=None)

# Select time range
start_time = '2022-09-01'
end_time = '2022-12-31'

# Europe and South Africa data slicing
sliced_hrest0_europe = full_hrest0.sel(time=slice(start_time, end_time),
                                      latitude=slice(35.00, 69.9),
                                      longitude=slice(-10.00, 39.9))

sliced_hrest0_sa = full_hrest0.sel(time=slice(start_time, end_time), 
                                   latitude=slice(-37.75, -22.00), 
                                   longitude=slice(15.25, 35.00))

sliced_era5_europe = full_era5.sel(time=slice(start_time, end_time),
                                   latitude=slice(69.9, 35.00),
                                      longitude=slice(-10.00, 39.9))

sliced_era5_sa = full_era5.sel(time=slice(start_time, end_time),
                               latitude=slice(-22.00, -37.75),
                               longitude=slice(15.25, 35.00))

# Constants
STATIC_VARIABLES = ["land_sea_mask", "soil_type", "geopotential_at_surface"]
surface_vars_names = ["2t", "10u", "10v", "msl"]
plots_titles = [
    "Two-meter temperature two steps forward prediction: RMSEs",
    "Ten-meter eastward wind speed two steps forward prediction: RMSEs",
    "Ten-meter southward wind speed two steps forward prediction: RMSEs",
    "Mean sea-level pressure two steps forward prediction: RMSEs"
]

selected_times = sliced_hrest0_europe.time

# Compute RMSE weights
europe_rmse_weights = rmse_weights(sliced_hrest0_europe.latitude, sliced_hrest0_europe.longitude)
sa_rmse_weights = rmse_weights(sliced_era5_sa.latitude, sliced_hrest0_sa.longitude)

# Initialize result storage
europe_rmses_list = {var: [] for var in surface_vars_names}
sa_rmses_list = {var: [] for var in surface_vars_names}
pred_dates_list = []

# Load model
model_initial = AuroraSmall(
    use_lora=False,
    autocast=True,
    stabilise_level_agg=True
)
model_initial.load_state_dict(torch.load('../model/aurora-0.25-small-pretrained.pth'))

# Main inference loop
for i in range(len(selected_times) - 3):
    europe_feature_hrest0_data = sliced_hrest0_europe.sel(time=slice(selected_times[i], selected_times[i+1]))
    europe_target_hrest0_data = sliced_hrest0_europe.sel(time=slice(selected_times[i+2], selected_times[i+3]))
    sa_feature_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
    sa_target_hrest0_data = sliced_hrest0_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))

    europe_feature_era5_data = sliced_era5_europe.sel(time=slice(selected_times[i], selected_times[i+1]))
    europe_target_era5_data = sliced_era5_europe.sel(time=slice(selected_times[i+2], selected_times[i+3]))
    sa_feature_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i], selected_times[i+1]))
    sa_target_era5_data = sliced_era5_sa.sel(time=slice(selected_times[i+2], selected_times[i+3]))

    # Extract features and targets
    europe_feature_surface_data, europe_target_surface_data = get_surface_feature_target_data(europe_feature_hrest0_data, europe_target_hrest0_data)
    europe_feature_atmos_data, europe_target_atmos_data = get_atmos_feature_target_data(europe_feature_hrest0_data, europe_target_hrest0_data)
    europe_feature_static_data, europe_target_static_data = get_static_feature_target_data(europe_feature_era5_data, europe_target_era5_data, STATIC_VARIABLES)
    
    sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
    sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
    sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data, STATIC_VARIABLES)

    # Create batches
    europe_feature_batch = create_hrest0_batch(europe_feature_surface_data, europe_feature_atmos_data, europe_feature_static_data)
    europe_target_batch = create_hrest0_batch(europe_target_surface_data, europe_target_atmos_data, europe_target_static_data)
    sa_feature_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
    sa_target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)
    # Predict
    sa_predictions = predict_fn(model=model_initial, batch=sa_feature_batch)
    europe_predictions = predict_fn(model=model_initial, batch=europe_feature_batch)
    # print(f"prediction: {europe_predictions}")
    
    # Compute RMSEs
    valid = True
    temp_europe_rmses = {}
    temp_sa_rmses = {}
    for var in surface_vars_names:
        europe_rmses, europe_pred_dates = rmse_fn(europe_predictions, europe_target_batch, var_name=var, weigths=europe_rmse_weights)
        sa_rmses, _ = rmse_fn(sa_predictions, sa_target_batch, var_name=var, weigths=sa_rmse_weights, area="sa")
        if np.isnan(europe_rmses).any():
            valid = False
            break
        temp_europe_rmses[var] = europe_rmses
        temp_sa_rmses[var] = sa_rmses

    if valid:
        pred_dates_list.append(europe_pred_dates)
        for var in surface_vars_names:
            europe_rmses_list[var].append(temp_europe_rmses[var])
            sa_rmses_list[var].append(temp_sa_rmses[var])

    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}")

# Save filtered RMSEs to CSV
# dict_clean = {"dates": pred_dates_list, "rmses": europe_rmses_list["2t"]}
# df_rmses = pd.DataFrame(dict_clean)
# df_rmses.to_csv("rmses_europe.csv")

# Plot cleaned RMSEs
for var, title in zip(surface_vars_names, plots_titles):
    plot_rmses(
        var,
        europe_rmses_list[var],
        sa_rmses_list[var],
        figsize=(12, 8),
        fontsize=18,
        date_ranges=pred_dates_list,
        title=title,
        save_path="../report/hrest0_europe",
        place = "Europe",
        atmos_level=None
    )
    print(f"Plot for {var} completed")

print("All plots done")