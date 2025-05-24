import xarray as xr
import torch
import os
import logging
import sys
import torch.optim as optim
from datetime import datetime
import numpy as np

from aurora import AuroraSmall, Batch, Metadata, rollout
from torch.utils.data import Dataset
from loss import AuroraLoss
from utils import (
    get_surface_feature_target_data,
    get_atmos_feature_target_data,
    get_static_feature_target_data,
    create_batch,
    create_hrest0_batch,
    predict_train_fn,
    predict_fn
)
from evaluation_metric import evaluation_rmse
log_file = "rmses_grid_sp_sa_vs_usa_eu.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


#
SURFACE_WEIGHTS = {"2t":[], "10u":[], "10v":[], "msl":[]} 

SURFACE_VARIABLES =["2t", "10u", "10v", "msl"] 


ATMOSPHERIC_VARIABLES = ["z", "q", "t", "u", "v"]





def evaluation_between_regions(
            model,
            era5_data=None, 
            hres_data=None,
            target_region=None,
            base_region= None,
            rollouts_num=8):
    
    tr_surface_rmses = {var:np.zeros(8) for var in SURFACE_VARIABLES}
    br_surface_rmses = {var:np.zeros(8) for var in SURFACE_VARIABLES}

    tr_atmospheric_rmses  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}
    br_atmospheric_rmses  = {var:np.zeros((13,8)) for var in ATMOSPHERIC_VARIABLES}



    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU if available

    counter = 0
    # regions
    
    target_region_lat_min ,target_region_lat_max , target_region_lon_min, target_region_lon_max = target_region
    base_region_lat_min ,base_region_lat_max , base_region_lon_min, base_region_lon_max = base_region
    
 

    for i in range(num_samples):
        counter+=1
        t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+2], selected_times[i+10]

        # target region data
        tr_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(target_region_lat_min, target_region_lat_max),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        tr_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(target_region_lat_max, target_region_lat_min),
                               longitude=slice(target_region_lon_min, target_region_lon_max))
        
        
        # get data base region
        
        br_feature_hres_data = hres_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        br_target_hres_data = hres_data.sel(time=slice(t2, t3),
                              latitude=slice(base_region_lat_min, base_region_lat_max),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_feature_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        br_target_era_data = era5_data.sel(time=slice(t0, t1),
                              latitude=slice(base_region_lat_max, base_region_lat_min),
                               longitude=slice(base_region_lon_min, base_region_lon_max))
        
        
        target_times = tr_target_hres_data.time

        for i in range(rollouts_num):
            # get target time
            # get only two data for target
            # Lead time target
            tr_lead_time_target = tr_target_hres_data.sel(time=slice(target_times[i], target_times[i+1]))
            # Lead time target
            br_lead_time_target = br_target_hres_data.sel(time=slice(target_times[i], target_times[i+1]))
            
            
            # Extract features and targets for target region
            tr_feature_surface_data, tr_target_surface_data = get_surface_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_atmos_data, tr_target_atmos_data = get_atmos_feature_target_data(tr_feature_hres_data, tr_lead_time_target)
            tr_feature_static_data, tr_target_static_data = get_static_feature_target_data(tr_feature_era_data, tr_target_era_data)

            # Extract features and targets for base region
            br_feature_surface_data, br_target_surface_data = get_surface_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_atmos_data, br_target_atmos_data = get_atmos_feature_target_data(br_feature_hres_data, br_lead_time_target)
            br_feature_static_data, br_target_static_data = get_static_feature_target_data(br_feature_era_data, br_target_era_data)

            
            
            # Create input and target batches for target regio
            
            tr_input_batch = create_hrest0_batch(tr_feature_surface_data, tr_feature_atmos_data, tr_feature_static_data).to(device)
            tr_target_batch = create_hrest0_batch(tr_target_surface_data, tr_target_atmos_data, tr_target_static_data).to(device)

            # Create input and target batches for target regio
            br_input_batch = create_hrest0_batch(br_feature_surface_data, br_feature_atmos_data, br_feature_static_data).to(device)
            br_target_batch = create_hrest0_batch(br_target_surface_data, br_target_atmos_data, br_target_static_data).to(device)

            
            # predictions
            tr_predictions =  predict_fn(model=model, batch=tr_input_batch, rollout_nums=rollouts_num)
            br_predictions =  predict_fn(model=model, batch=br_input_batch, rollout_nums=rollouts_num)
            
            # Compuete rmse for all lead times
            tr_prediction = tr_predictions[i]
            br_prediction = br_predictions[i]
            
            for surf_var in SURFACE_VARIABLES:
                # for target region
                tr_pred_tensor = tr_prediction.surf_vars[surf_var].squeeze()
                tr_pred_tensor = tr_pred_tensor.to("cuda")
                
                # for target region
                br_pred_tensor = br_prediction.surf_vars[surf_var].squeeze()
                br_pred_tensor = br_pred_tensor.to("cuda")
              
                # target tensors
                tr_target_tensor = tr_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                tr_target_tensor = tr_target_tensor.to("cuda")
                
                br_target_tensor = br_target_batch.surf_vars[surf_var].squeeze()[0,:,:]
                br_target_tensor = br_target_tensor.to("cuda")
                
                # Rmses
                tr_surface_rmses[surf_var][i]+= evaluation_rmse(
                                                    tr_target_tensor, 
                                                    tr_pred_tensor, 
                                                    torch.tensor(tr_feature_hres_data.latitude.values),
                                                    torch.tensor(tr_feature_hres_data.longitude.values)   
                                                ) 
                br_surface_rmses[surf_var][i]+= evaluation_rmse(
                                                    br_target_tensor, 
                                                    br_pred_tensor, 
                                                    torch.tensor(br_feature_hres_data.latitude.values),
                                                    torch.tensor(br_feature_hres_data.longitude.values)   
                                                ) 
                
            ## Atmospheric
            atmos_levels_num = tr_atmospheric_rmses["z"].shape[0]
            for c in range(atmos_levels_num):
                for atmos_var in ATMOSPHERIC_VARIABLES:
                    #target region
                    tr_pred_tensor = tr_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    tr_pred_tensor = tr_pred_tensor.to("cuda")
                    
                    #target region
                    br_pred_tensor = br_prediction.atmos_vars[atmos_var].squeeze()[c,:,:]
                    br_pred_tensor = br_pred_tensor.to("cuda")
                    
                    
                    tr_target_tensor = tr_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    tr_target_tensor = tr_target_tensor.to("cuda")
                    
                    br_target_tensor = br_target_batch.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                    br_target_tensor = br_target_tensor.to("cuda")
                    
                    
                    
                    #Atmospheric rms
                    tr_atmospheric_rmses[atmos_var][c, i] += evaluation_rmse(
                                                    tr_target_tensor, 
                                                    tr_pred_tensor, 
                                                    torch.tensor(tr_feature_hres_data.latitude.values),
                                                    torch.tensor(tr_feature_hres_data.longitude.values)    
                                                ) 
                    br_atmospheric_rmses[atmos_var][c, i] += evaluation_rmse(
                                                    br_target_tensor, 
                                                    br_pred_tensor, 
                                                    torch.tensor(br_feature_hres_data.latitude.values),
                                                    torch.tensor(br_feature_hres_data.longitude.values)    
                                                ) 
                 
        if not counter%10:
            logger.info(f"Iteration {counter} done")
    logger.info("Evaluation completed")
    
    return {
        'counter': counter,
        'target_region_surface_rmses': tr_surface_rmses,
        'target_region_atmospheric_rmses': tr_atmospheric_rmses,
        'base_region_surface_rmses': br_surface_rmses,
        'base_region_atmospheric_rmses': br_atmospheric_rmses
    }
