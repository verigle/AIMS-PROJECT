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





ERA5_SURFACE_VARIABLES = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']

ERA5_ATMOSPHERIC_VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]

ERA5_STATIC_VARIABLES = ["land_sea_mask", "soil_type", "geopotential_at_surface"]



class ERA5ZarrDataset(Dataset):
    def __init__(self, era5_sliced_sa,
                 surface_vars=ERA5_SURFACE_VARIABLES, 
                 atmos_vars=ERA5_ATMOSPHERIC_VARIABLES,
                 static_vars = ERA5_STATIC_VARIABLES,
                 lead_time=1):
        self.surf_vars_ds = era5_sliced_sa[surface_vars]
        self.atmos_vars_ds = era5_sliced_sa[atmos_vars]
        self.static_vars_ds = era5_sliced_sa[static_vars]
        self.lead_time = lead_time
 
        
        self.time_indices = range(len(self.atmos_vars_ds.time - lead_time))
        # self.sequence_length = sequence_length
        # self.time_indices = range(sequence_length, len(surf_vars_ds.time) - 1)  # Ensure label is within bounds

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        i = self.time_indices[idx]

        # Input batch
        surf_vars_input = {
            "2t": torch.from_numpy(self.surf_vars_ds["2m_temperature"].values[i, i+1][None]),
            "10u": torch.from_numpy(self.surf_vars_ds["10m_u_component_of_wind"].values[i, i+1][None]),
            "10v": torch.from_numpy(self.surf_vars_ds["10m_v_component_of_wind"].values[i, i+1][None]),
            "msl": torch.from_numpy(self.surf_vars_ds["mean_sea_level_pressure"].values[i, i+1][None]),
        }

        static_vars = {
            "z": torch.from_numpy(self.static_vars_ds["geopotential_at_surface"].values),
            "slt": torch.from_numpy(self.static_vars_ds["soil_type"].values),
            "lsm": torch.from_numpy(self.static_vars_ds["land_sea_mask"].values),
        }

        atmos_vars_input = {
            "t": torch.from_numpy(self.atmos_vars_ds["temperature"].values[i, i+1][None]),
            "u": torch.from_numpy(self.atmos_vars_ds["u_component_of_wind"].values[i, i+1][None]),
            "v": torch.from_numpy(self.atmos_vars_ds["v_component_of_wind"].values[i, i+1][None]),
            "q": torch.from_numpy(self.atmos_vars_ds["specific_humidity"].values[i, i+1][None]),
            "z": torch.from_numpy(self.atmos_vars_ds["geopotential"].values[i, i+1][None]),
        }

        metadata = Metadata(
            lat=torch.from_numpy(self.surf_vars_ds.latitude.values),
            lon=torch.from_numpy(self.surf_vars_ds.longitude.values),
            time=self.surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i+1],
            atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.level.values)
        )

        input = Batch(surf_vars=surf_vars_input, static_vars=static_vars, atmos_vars=atmos_vars_input, metadata=metadata)
        
        # Label (next time step)
        surf_vars_label = {
            "2t": torch.from_numpy(self.surf_vars_ds["2m_temperature"].values[i+2:i+2+self.lead_time][None]),
            "10u": torch.from_numpy(self.surf_vars_ds["10m_u_component_of_wind"].values[i+2:i+2+self.lead_time][None]),
            "10v": torch.from_numpy(self.surf_vars_ds["10m_v_component_of_wind"].values[i+2:i+2+self.lead_time][None]),
            "msl": torch.from_numpy(self.surf_vars_ds["mean_sea_level_pressure"].values[i+2:i+2+self.lead_time][None]),
        }

        atmos_vars_label = {
            "t": torch.from_numpy(self.atmos_vars_ds["temperature"].values[i+2:i+2+self.lead_time][None]),
            "u": torch.from_numpy(self.atmos_vars_ds["u_component_of_wind"].values[i+2:i+2+self.lead_time][None]),
            "v": torch.from_numpy(self.atmos_vars_ds["v_component_of_wind"].values[i+2:i+2+self.lead_time][None]),
            "q": torch.from_numpy(self.atmos_vars_ds["specific_humidity"].values[i+2:i+2+self.lead_time][None]),
            "z": torch.from_numpy(self.atmos_vars_ds["geopotential"].values[i+2:i+2+self.lead_time][None]),
        }
        
        metadata_label = Metadata(
            lat=torch.from_numpy(self.surf_vars_ds.latitude.values),
            lon=torch.from_numpy(self.surf_vars_ds.longitude.values),
            time=self.surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i+2],
            atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.level.values)
        )
        
        label = Batch(surf_vars=surf_vars_label, static_vars=static_vars, atmos_vars=atmos_vars_label, metadata=metadata_label)

        return input, label
