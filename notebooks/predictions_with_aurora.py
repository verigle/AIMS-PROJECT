
# # Downloading the Data

# pip uninstall torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 

# In[ ]:



from pathlib import Path

import cdsapi

# Data will be downloaded here.
download_path = Path("data")

c = cdsapi.Client()

download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# Download the static variables.
if not (download_path / "static.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "static.nc"),
    )
print("Static variables downloaded!")

# Download the surface-level variables.
if not (download_path / "2023-01-01-surface-level.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2023-01-01-surface-level.nc"),
    )
print("Surface-level variables downloaded!")

# Download the atmospheric variables.
if not (download_path / "2023-01-01-atmospheric.nc").exists():
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ],
            "pressure_level": [
                "50",
                "100",
                "150",
                "200",
                "250",
                "300",
                "400",
                "500",
                "600",
                "700",
                "850",
                "925",
                "1000",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2023-01-01-atmospheric.nc"),
    )
print("Atmospheric variables downloaded!")


# # Preparing a Batch

# In[ ]:


import torch
import xarray as xr

from aurora import Batch, Metadata

print("creating Batch")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")
print("done")
i = 1  # Select this time index in the downloaded data.

# batch = Batch(
#     surf_vars={
#         # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
#         # batch dimension of size one.
#         "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
#         "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
#         "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
#         "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
#     },
#     static_vars={
#         # The static variables are constant, so we just get them for the first time.
#         "z": torch.from_numpy(static_vars_ds["z"].values[0]),
#         "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
#         "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
#     },
#     atmos_vars={
#         "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
#         "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
#         "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
#         "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
#         "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
#     },
#     metadata=Metadata(
#         lat=torch.from_numpy(surf_vars_ds.latitude.values),
#         lon=torch.from_numpy(surf_vars_ds.longitude.values),
#         # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
#         # `datetime.datetime`s. Note that this needs to be a tuple of length one:
#         # one value for every batch element.
#         time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
#         atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
#     ),
# )


# Reduce batch size and use only smaller slices for debugging
batch = Batch(
    surf_vars={
        # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
        # batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[max(0, i - 1):i + 1][None]),  # Slice data smaller for debugging
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[max(0, i - 1):i + 1][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[max(0, i - 1):i + 1][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[max(0, i - 1):i + 1][None]),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[max(0, i - 1):i + 1][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[max(0, i - 1):i + 1][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[max(0, i - 1):i + 1][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[max(0, i - 1):i + 1][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[max(0, i - 1):i + 1][None]),
    },
    metadata=Metadata(
        # Slice the latitude and longitude to reduce memory usage
        lat=torch.from_numpy(surf_vars_ds.latitude.values[:100]),  # Only load a small subset
        lon=torch.from_numpy(surf_vars_ds.longitude.values[:100]),  # Only load a small subset
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element.
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values[:5]),  # Limit the pressure levels
    ),
)

# After creating the batch, delete unneeded variables to free memory
del surf_vars_ds, static_vars_ds, atmos_vars_ds

# Print memory usage after batch creation for debugging
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")  # If using a GPU
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e6} MB")    # If using a GPU



print("Batch created")
# 

# # Loading and Running the Model

# In[ ]:

print("Loading the weights")

from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout

# model = AuroraSmall()

# model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

# print("Weights Loaded")

# In[ ]:


# model.eval()
# model = model.to("cpu")

# with torch.inference_mode():
#     preds = [pred.to("cpu") for pred in rollout(model, batch, steps=1)]

# model = model.to("cpu")


# In[17]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    # pred = preds[i]

    # ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    # ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    # if i == 0:
    #     ax[i, 0].set_title("Aurora Prediction")
    # ax[i, 0].set_xticks([])
    # ax[i, 0].set_yticks([])

    ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("ERA5")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.show()



