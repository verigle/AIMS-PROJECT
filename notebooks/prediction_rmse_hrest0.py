#!/usr/bin/env python
# coding: utf-8

# In[2]:

import gc
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


# # Load the model

# In[3]:


model = AuroraSmall()

model.load_state_dict(torch.load('../model/aurora.pth'))


# # Data

# ## World

# In[4]:


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store, consolidated=True, chunks=None)


# ### Subset data from 2022

# In[5]:


start_time = '2022-06-01'
end_time = '2022-12-31'

sliced_hrest0_world = (
    full_hrest0
    .sel(time=slice(start_time, end_time))
    .isel(time=slice(None, -2))
)


# In[6]:


target_sliced_hrest0_world = (
    full_hrest0
    .sel(time=slice(start_time, end_time))  # Select the time range
    .isel(time=slice(2, None))  # Skip the first two time steps
)


# # Era 5 data for static variables

# In[7]:


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)


# In[8]:


start_time = '2022-06-01'
end_time = '2022-12-31'
data_inner_steps = 6  

sliced_era5_world = (
    full_era5
    .sel(time=slice(start_time, end_time))
    .isel(time=slice(None, -2))
)


# In[9]:


target_sliced_era5_world = (
    full_era5
    .sel(time=slice(start_time, end_time))  # Select the time range
    .isel(time=slice(2, None))  # Skip the first two time steps
)


# ### Surface variables

# In[10]:


# List of surface variable names
surface_vars = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']

# Select surface variables
surf_vars_ds = sliced_hrest0_world[surface_vars]
target_surf_vars_ds = target_sliced_hrest0_world[surface_vars]


# ### Atmospherique variables

# In[11]:


atmostpheric_variables = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
atmos_vars_ds = sliced_hrest0_world[atmostpheric_variables]
target_atmos_vars_ds = target_sliced_hrest0_world[atmostpheric_variables]


# ## Static variables

# In[12]:


static_variables = ["land_sea_mask", "soil_type", "geopotential_at_surface"]
static_vars_ds = sliced_era5_world[static_variables]
target_static_vars_ds = target_sliced_era5_world[static_variables]


# ## Create batches

# In[31]:


def _prepare(x: np.ndarray, i) -> torch.Tensor:
    """Prepare a variable.

    This does the following things:
    * Select time indices `i` and `i - 1`.
    * Insert an empty batch dimension with `[None]`.
    * Flip along the latitude axis to ensure that the latitudes are decreasing.
    * Copy the data, because the data must be contiguous when converting to PyTorch.
    * Convert to PyTorch.
    """
    return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy())


# In[62]:

from torch.utils.data import DataLoader

class ERA5ZarrDataset(Dataset):
    def __init__(self, surf_vars_ds, atmos_vars_ds, static_vars_ds, sequence_length):
        self.surf_vars_ds = surf_vars_ds
        self.atmos_vars_ds = atmos_vars_ds
        self.static_vars_ds = static_vars_ds
        self.sequence_length = sequence_length
        self.time_indices = range(sequence_length, len(surf_vars_ds.time))

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        i = self.time_indices[idx]

        surf_vars = {
            "2t": torch.tensor(self.surf_vars_ds["2m_temperature"].values[[i - 1, i]][None], dtype=torch.float32),
            "10u": torch.tensor(self.surf_vars_ds["10m_u_component_of_wind"].values[[i - 1, i]][None], dtype=torch.float32),
            "10v": torch.tensor(self.surf_vars_ds["10m_v_component_of_wind"].values[[i - 1, i]][None], dtype=torch.float32),
            "msl": torch.tensor(self.surf_vars_ds["mean_sea_level_pressure"].values[[i - 1, i]][None], dtype=torch.float32),
        }

        static_vars = {
            "z": torch.tensor(self.static_vars_ds["geopotential_at_surface"].values, dtype=torch.float32),
            "slt": torch.tensor(self.static_vars_ds["soil_type"].values, dtype=torch.float32),
            "lsm": torch.tensor(self.static_vars_ds["land_sea_mask"].values, dtype=torch.float32),
        }

        atmos_vars = {
            "t": torch.tensor(self.atmos_vars_ds["temperature"].values[[i - 1, i]][None], dtype=torch.float32),
            "u": torch.tensor(self.atmos_vars_ds["u_component_of_wind"].values[[i - 1, i]][None], dtype=torch.float32),
            "v": torch.tensor(self.atmos_vars_ds["v_component_of_wind"].values[[i - 1, i]][None], dtype=torch.float32),
            "q": torch.tensor(self.atmos_vars_ds["specific_humidity"].values[[i - 1, i]][None], dtype=torch.float32),
            "z": torch.tensor(self.atmos_vars_ds["geopotential"].values[[i - 1, i]][None], dtype=torch.float32),
        }

        metadata = Metadata(
            lat=torch.tensor(self.surf_vars_ds.latitude.values, dtype=torch.float32),
            lon=torch.tensor(self.surf_vars_ds.longitude.values, dtype=torch.float32),
            time=(self.surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.level.values),
        )

        return Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)


# In[63]:


world_batches = ERA5ZarrDataset(surf_vars_ds, atmos_vars_ds, static_vars_ds,1)
target_world_batches = ERA5ZarrDataset(target_surf_vars_ds, target_atmos_vars_ds, target_static_vars_ds,1)

batch_size = 1  # Set a reasonable batch size for memory efficiency

world_loader = DataLoader(world_batches, batch_size=batch_size, shuffle=False)
target_loader = DataLoader(target_world_batches, batch_size=batch_size, shuffle=False)


# ### South Africa Data

# In[64]:


start_time = '2022-06-01'
end_time = '2022-12-31'

lat_max = -22.00 
lat_min = -37.75  

lon_min = 15.25   
lon_max = 35.00   

sliced_hrest0_SA = (
    full_hrest0
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )
    .isel(time=slice(None, -2))
)

target_sliced_hrest0_SA = (
    full_hrest0
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )  
    .isel(time=slice(2, None))  # Skip the first two time steps
)


# In[65]:


start_time = '2022-06-01'
end_time = '2022-12-31'

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
    .isel(time=slice(None, -2))
)

target_sliced_era5_SA = (
    full_era5
    .sel(
        time=slice(start_time, end_time),
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)  
    )  
    .isel(time=slice(2, None))  # Skip the first two time steps
)


# In[66]:


surf_vars_ds_SA = sliced_hrest0_SA[surface_vars]

target_surf_vars_ds_SA = target_sliced_hrest0_SA[surface_vars]

atmos_vars_ds_SA = sliced_hrest0_SA[atmostpheric_variables]

target_atmos_vars_ds_SA = target_sliced_hrest0_SA[atmostpheric_variables]

static_vars_ds_SA = sliced_era5_SA[static_variables]

target_static_vars_ds_SA = target_sliced_era5_SA[static_variables]


# In[67]:


SA_batches = ERA5ZarrDataset(surf_vars_ds_SA, atmos_vars_ds_SA, static_vars_ds_SA,1)
target_SA_batches = ERA5ZarrDataset(target_surf_vars_ds_SA, target_atmos_vars_ds_SA, target_static_vars_ds_SA,1)

batch_size = 1  # Set a reasonable batch size for memory efficiency

sa_world_loader = DataLoader(SA_batches, batch_size=batch_size, shuffle=False)
sa_target_loader = DataLoader(target_SA_batches, batch_size=batch_size, shuffle=False)

# ## Predictions Function

# In[68]:


def predict_fn(model, batch):
    model.eval()
    model = model.to("cuda")
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
    model = model.to("cpu")
    return preds


# # Custom RMSE function

# ## Grid weights

# In[69]:


def rmse_weights(latitudes, longitudes, R=6371.0):
    # convert to gradient
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    
    dlat = np.abs(np.diff(lat_rad).mean())  # Average latitude difference
    dlon = np.abs(np.diff(lon_rad).mean())  # Average longitude difference

    # Calculate the area for each latitude band
    areas = R**2 * dlon * np.abs(np.sin(lat_rad + dlat/2) - np.sin(lat_rad - dlat/2))

    # Expand areas to match the shape of the grid
    area_grid = np.outer(areas, np.ones(len(longitudes)))
    area_grid = area_grid/area_grid.sum()
    
    
    return area_grid
    



# ### world rmse weights

# In[70]:


# world_rmse_weights = rmse_weights(sliced_era5_world.latitude, sliced_era5_world.longitude, R=6371.0)


# ### South Africa rmse weights

# In[71]:


# SA_rmse_weights = rmse_weights(sliced_era5_SA.latitude, sliced_era5_SA.longitude, R=6371.0)


# In[72]:


def custom_rmse(actual, prediction, weigths):
    return (((actual-prediction)**2)*weigths).sum()


# # RMSEs World dataset

# In[73]:


def rmse_fn(model, feature_batch, target_batch, var_name, weigths=None, var_type="surface", atmos_level_idx=0):
    predictions = predict_fn(model, batch=feature_batch)
    two_steps_rmse = []
    pred_dates = []
    for i in range(len(predictions)):
        pred = predictions[i]
        if var_type=="surface":
            prediction = pred.surf_vars[var_name][0, 0].numpy()
            actual = target_batch.surf_vars[var_name].squeeze()[i,:,:][1:, :]
            # actual = target_batch.surf_vars[var_name][0, 0].numpy()
            
            rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            # rmse_ = custom_rmse(actual, prediction, weigths[1:,:])
            # print(rmse1)
            # two_steps_rmse.append(rmse_.item())
            two_steps_rmse.append(rmse)
            pred_dates.append(pred.metadata.time[0])
        # Atmospherique variable
        elif var_type=="atmosphere":
            prediction = pred.atmos_vars[var_name].squeeze()[atmos_level_idx,:,:].numpy().squeeze()
            # actual = target_batch.atmos_vars[var_name].squeeze()[i,:,:][1:, :]
            actual = target_batch.atmos_vars[var_name].squeeze()[i,atmos_level_idx,:,:].numpy()[:-1,:]
            # rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            rmse_ = custom_rmse(actual, prediction, weigths[1:,:])
            two_steps_rmse.append(rmse_.item())
            pred_dates.append(pred.metadata.time[0])
    return two_steps_rmse, pred_dates


# # RMSEs South Africa dataset

# In[74]:


def rmse_fn_sa(model, actual_batch, target_batch, var_name, weigths=None, var_type="surface",  atmos_level_idx=0):
    predictions = predict_fn(model, batch=actual_batch)
    two_steps_rmse = []
    pred_dates = []
    for i in range(len(predictions)):
        pred = predictions[i]
        if var_type=="surface":
            # prediction = pred.surf_vars[var_name][0, 0].numpy()
            # actual = actual_batch.surf_vars[var_name][0, 0].numpy()
            prediction = pred.surf_vars[var_name][0, 0].numpy()
            actual = target_batch.surf_vars[var_name].squeeze()[i,:,:]
            rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            
            # rmse1 = rmse(actual, prediction)
            # computed_rmse = custom_rmse(actual, prediction, weigths)
            # two_steps_rmse.append(computed_rmse.item())
            two_steps_rmse.append(rmse)
            pred_dates.append(pred.metadata.time[0])
            # print(computed_rmse.item())
        elif var_type=="atmosphere":
            prediction = pred.atmos_vars[var_name].squeeze()[atmos_level_idx,:,:].numpy().squeeze()
            actual = target_batch.atmos_vars[var_name].squeeze()[i,atmos_level_idx,:,:].numpy()
            # rmse = root_mean_squared_error(actual.flatten(), prediction.flatten())
            rmse_ = custom_rmse(actual, prediction, weigths)
            two_steps_rmse.append(rmse_.item())
            pred_dates.append(pred.metadata.time[0])
    return two_steps_rmse, pred_dates


# # PLot RMSES

# In[75]:


def plot_rmses(variable, rmses_world, rmses_sa, 
               figsize=(12, 8), fontsize=18,
               date_ranges=None, 
               title="Two Steps Forward Prediction: RMSEs",
               save_path="../report/hrest0",
               atmos_level=None):

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Extract dates
    date_times_6_hours = [date1 for date1, date2 in date_ranges]
    date_times_12_hours = [date2 for date1, date2 in date_ranges]
    formatted_dates_6_hours = [dt.strftime('%Y-%m-%d (%H:%M)') for dt in date_times_6_hours]
    formatted_dates_12_hours = [dt.strftime('%Y-%m-%d (%H:%M)') for dt in date_times_12_hours]

    # Convert x-axis to indices
    x_indices = np.arange(len(formatted_dates_6_hours))

    # Select a subset of dates for x-axis labels
    num_ticks = min(6, len(formatted_dates_6_hours))
    tick_positions = np.linspace(0, len(formatted_dates_6_hours) - 1, num_ticks, dtype=int)

    # Plot RMSEs with improved colors and styles
    ax.plot(x_indices, np.array(rmses_world)[:, 0], label="Global RMSE (6h Forecast)", color="blue", linestyle="-", linewidth=2)
    ax.plot(x_indices, np.array(rmses_sa)[:, 0], label="South Africa RMSE (6h Forecast)", color="orange", linestyle="-", linewidth=2)
    ax.plot(x_indices, np.array(rmses_world)[:, 1], label="Global RMSE (12h Forecast)", color="blue", linestyle="--", linewidth=2)
    ax.plot(x_indices, np.array(rmses_sa)[:, 1], label="South Africa RMSE (12h Forecast)", color="orange", linestyle="--", linewidth=2)

    # Set selected x-ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([formatted_dates_12_hours[i] for i in tick_positions], rotation=30, ha='right')

    # Improve legend appearance
    ax.legend(title="Forecast Horizon", title_fontsize=fontsize-2, fontsize=fontsize-4,
              bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # Improve axis labels and title
    ax.set_xlabel("Forecast Date", fontsize=fontsize-2)
    ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=fontsize-2)
    ax.set_title(title, fontsize=fontsize, pad=20)
    if atmos_level:
        # Save the plots
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.png", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}-{atmos_level}.svg", bbox_inches="tight")
    else:
        plt.savefig(f"{save_path}/rmse-{variable}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/rmse-{variable}.png", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}/rmse-{variable}.svg", bbox_inches="tight")

    plt.show()


# # Surface Variables

# ## Two-meter temperature in K: 2t

# In[ ]:

print("Starting for the world")
rmses_world_2t = []
dates_world_2t = []

for feature_batch, target_batch in zip(world_loader, target_loader):
    feature_batch = feature_batch.to("cuda")
    target_batch = target_batch.to("cuda")

    rmse, date = rmse_fn(model, feature_batch=feature_batch, target_batch=target_batch, var_name="2t", var_type="surface")
    rmses_world_2t.append(rmse)
    dates_world_2t.append(date)

    # Free memory
    del feature_batch, target_batch, rmse, date
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:
print("Starting for South Africa")

rmses_SA_2t = []
dates_SA_2t = []

for feature_batch, target_batch in zip(sa_world_loader, sa_target_loader):
    feature_batch = feature_batch.to("cuda")
    target_batch = target_batch.to("cuda")

    rmse, date = rmse_fn_sa(model, actual_batch=feature_batch, target_batch=target_batch, var_name="2t", var_type="surface")
    rmses_SA_2t.append(rmse)
    dates_SA_2t.append(date)

    # Free memory
    del feature_batch, target_batch, rmse, date
    torch.cuda.empty_cache()
    gc.collect()



plot_rmses("2t",rmses_world_2t, rmses_SA_2t, 
            figsize=(15, 8), fontsize=18,
            date_ranges=dates_world_2t, title="Two-meter temperature two steps forward prediction: RMSES")

