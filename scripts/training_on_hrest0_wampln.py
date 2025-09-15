#!/usr/bin/env python
# coding: utf-8


import xarray as xr
from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout
import matplotlib.pyplot as plt

from pathlib import Path

import cdsapi
import numpy as np
from sklearn.metrics import root_mean_squared_error
import gcsfs

from torch.utils.data import Dataset
from aurora import Batch, Metadata
import os

import sys
import torch.optim.lr_scheduler as lr_scheduler
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, custom_rmse
from lora import  create_custom_model, print_trainable_parameters, full_linear_layer_lora

from train_hres import training, logger
import torch.optim as optim
from loss import AuroraLoss




model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
)
# model.load_state_dict(torch.load('../model/urora-0.25-small-pretrained1.pth'))

# model = AuroraSmall()
# model = AuroraSmall(
#     use_lora=False,  # model was not fine-tuned.
#     autocast=True,  # Use AMP.
#     stabilise_level_agg=True
# )
model = full_linear_layer_lora(model, lora_r = 16, lora_alpha = 4)
checkpoint = torch.load('../model/training/hrest0/wampln/checkpoint_epoch_5.pth')
print("Loading model from checkpoint")

model.load_state_dict(checkpoint['model_state_dict'])


message  = print_trainable_parameters(model)

logger.info(message)


# # Get south africa Data

fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)



# start_time, end_time = '2022-11-01', '2023-01-31'
start_time, end_time = '2015-01-01', '2018-12-31' #'2021-12-31'
# start_time, end_time = '2023-01-08', '2023-01-31'



lat_max = -22.00 
lat_min = -37.75  

lon_min = 15.25   
lon_max = 35.00   
sliced_era5_SA = (
    full_era5
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )
)

################################"" get hres data
store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)
sliced_hrest0_sa = full_hrest0.sel(time=slice(start_time, end_time), 
                                   latitude=slice(lat_min, lat_max), 
                                   longitude=slice(lon_min, lon_max))


optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


criterion = AuroraLoss()




model,  maps = training(model=model, criterion=criterion,
             num_epochs=100, optimizer=optimizer,
             era5_data=sliced_era5_SA, 
             hres_data=sliced_hrest0_sa,
             dataset_name="OTHER", lr_scheduler=scheduler,
             accumulation_steps=8)


torch.save(model.state_dict(), "..model/training/hrest0/model7/best_hres_model.pth")
save_path  = "../report/training/hres/model7"
_, ax = plt.subplots(figsize=(8, 6), dpi=300)

ax.plot(np.arange(1,len(maps)+1), maps)
ax.set_ylabel("Mean Absolute Error (MAP)")
ax.set_xlabel("Epoch")
plt.savefig(f"{save_path}/map-learning-curve.pdf", bbox_inches="tight")
plt.savefig(f"{save_path}/map-learning-curve.png", bbox_inches="tight")
plt.savefig(f"{save_path}/map-learning-curve.svg", bbox_inches="tight")
