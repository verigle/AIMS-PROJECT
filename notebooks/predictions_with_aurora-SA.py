from pathlib import Path
import cdsapi
import xarray as xr
import torch
from aurora import Batch, Metadata, AuroraSmall, rollout

# Define South Africa's geographical bounds
south_africa_bounds = {
    "north": -22.0,  # Northernmost latitude
    "south": -35.0,  # Southernmost latitude
    "west": 16.0,    # Westernmost longitude
    "east": 33.0     # Easternmost longitude
}

# Data will be downloaded here.
download_path = Path("data")
download_path.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

# Download static variables
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
        "area": [
            south_africa_bounds["north"],
            south_africa_bounds["west"],
            south_africa_bounds["south"],
            south_africa_bounds["east"]
        ]
    },
    str(download_path / "static-south-africa.nc"),
)
print("Static variables downloaded for South Africa!")

# Download surface-level variables
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
        "area": [
            south_africa_bounds["north"],
            south_africa_bounds["west"],
            south_africa_bounds["south"],
            south_africa_bounds["east"]
        ]
    },
    str(download_path / "surface-south-africa.nc"),
)
print("Surface-level variables downloaded for South Africa!")

# Download atmospheric variables
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
            "50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"
        ],
        "year": "2023",
        "month": "01",
        "day": "01",
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "format": "netcdf",
        "area": [
            south_africa_bounds["north"],
            south_africa_bounds["west"],
            south_africa_bounds["south"],
            south_africa_bounds["east"]
        ]
    },
    str(download_path / "atmospheric-south-africa.nc"),
)
print("Atmospheric variables downloaded for South Africa!")

# Load and filter datasets
static_vars_ds = xr.open_dataset(download_path / "static-south-africa.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "surface-south-africa.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "atmospheric-south-africa.nc", engine="netcdf4")

surf_vars_ds = surf_vars_ds.sel(latitude=slice(south_africa_bounds["north"], south_africa_bounds["south"]),
                                longitude=slice(south_africa_bounds["west"], south_africa_bounds["east"]))
atmos_vars_ds = atmos_vars_ds.sel(latitude=slice(south_africa_bounds["north"], south_africa_bounds["south"]),
                                  longitude=slice(south_africa_bounds["west"], south_africa_bounds["east"]))

# Prepare batch for Aurora model
i = 1  # Select time index
batch = Batch(
    surf_vars={
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
    },
    static_vars={
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

# Load and run Aurora model
model = AuroraSmall()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.eval()
with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

model = model.to("cpu")


# In[17]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("ERA5")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()





