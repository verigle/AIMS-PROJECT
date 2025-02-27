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


# select time range
start_time = '2022-12-31'
end_time = '2023-01-31'

# world
sliced_era5_world = (
    full_era5
    .sel(time=slice(start_time, end_time))
)

# South Africa


lat_max = -22.00 
lat_min = -37.75  

lon_min = 15.25   
lon_max = 35.00   

sliced_era5_sa = (
    full_era5
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )
)

############Surface variables#################################
surface_vars_names=["2t", "10u", "10v", "msl"]
plots_titles = ["Two-meter temperature two steps forward prediction: RMSES",
                "Ten-meter eastward wind speed two steps forward prediction: RMSES",
                "Ten-meter southward wind speed two steps forward prediction: RMSES",
                "Mean sea-level pressure two steps forward prediction: RMSES"]

selected_times=sliced_era5_world.time

world_latitudes = sliced_era5_world.latitude
world_longitudes = sliced_era5_world.longitude

sa_latitudes = sliced_era5_sa.latitude
sa_longitudes = sliced_era5_sa.longitude

world_rmse_weights = rmse_weights(world_latitudes, world_longitudes)[1:,:]
sa_rmse_weights = rmse_weights(sa_latitudes, sa_longitudes)
print("Data processing done")

################### Main part#################################################################################
for var, title in zip(surface_vars_names, plots_titles):
    
    world_rmses_list=[]; pred_dates_list=[]
    sa_rmses_list=[]
    iterations = 0
    for i in range(0, len(sliced_era5_world.time)-3):
        iterations+=1
        # get current and previous time step data
        world_feature_data =  (
                sliced_era5_world
                .sel(time=slice(selected_times[i], selected_times[i+1]))
            )
        sa_feature_data =  (
                sliced_era5_sa
                .sel(time=slice(selected_times[i], selected_times[i+1]))
            )
        # get  the next to timz step data
        world_target_data =  (
                sliced_era5_world
                .sel(time=slice(selected_times[i+2], selected_times[i+3]))
            )
        sa_target_data =  (
                sliced_era5_sa
                .sel(time=slice(selected_times[i+2], selected_times[i+3]))
            )
        
        # get each type of data(surface, static atmosphere)
        world_feature_surface_data, world_target_surface_data = get_surface_feature_target_data(world_feature_data, world_target_data)
        world_feature_atmos_data, world_target_atmos_data = get_atmos_feature_target_data(world_feature_data, world_target_data)
        world_feature_static_data, world_target_static_data = get_static_feature_target_data(world_feature_data, world_target_data)
        
        sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_data, sa_target_data)
        sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_data, sa_target_data)
        sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_data, sa_target_data)
        
        # create batch for each of them
        world_feature_bacth =  create_batch(world_feature_surface_data, world_feature_atmos_data, world_feature_static_data)
        world_target_bacth = create_batch(world_target_surface_data, world_target_atmos_data, world_target_static_data)
        
        sa_feature_bacth =  create_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
        sa_target_bacth = create_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)
        # get prediction
        world_predictions = predict_fn(batch=world_feature_bacth)
        sa_predictions = predict_fn(batch=sa_feature_bacth)
        # compute the rmse
        world_rmses, world_pred_dates = rmse_fn(predictions=world_predictions, 
                target_batch=world_target_bacth, var_name=var,
                weigths=world_rmse_weights)
        
        sa_rmses, sa_pred_dates = rmse_fn(predictions=sa_predictions, 
                target_batch=sa_target_bacth, var_name=var,
                weigths=sa_rmse_weights, area="sa")
        # append result to the list
        world_rmses_list.append(world_rmses); pred_dates_list.append(world_pred_dates)
        sa_rmses_list.append(sa_rmses)
        if iterations%10==0:
            print(f"Iterations {iterations} for {var}")
        
    plot_rmses(var, world_rmses_list, sa_rmses_list, 
        figsize=(12, 8), fontsize=18,
        date_ranges=pred_dates_list, 
        title=title,
        save_path="../report/era5",
        atmos_level=None)
    print("Plot Done")
    