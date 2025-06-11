

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
from matplotlib.colors import TwoSlopeNorm


import seaborn as sns



import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter  # <-- Add this import



import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, create_hrest0_batch


# In[78]:


from evaluation_three_regions import evaluation_between_regions
from lora import create_custom_model, full_linear_layer_lora

torch.use_deterministic_algorithms(True)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.8,
    'figure.dpi': 150,
})


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)



start_time, end_time = '2022-01-01', '2022-12-31' 

  
sliced_era5 = (
    full_era5
    .sel(
        time=slice(start_time, end_time) 
    )
)

store_hrest0 = fs.get_mapper('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
full_hrest0 = xr.open_zarr(store=store_hrest0, consolidated=True, chunks=None)
sliced_hrest0 = full_hrest0.sel(time=slice(start_time, end_time))

model = AuroraSmall(
    use_lora=False,  # fine_tuned_Model was not fine-tuned.
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = full_linear_layer_lora(model, lora_r = 16, lora_alpha = 4)
checkpoint = torch.load('../model/training/hrest0/wampln/checkpoint_epoch_3.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])



USA_REGION = (
      20.00,  # Southern boundary (approximate)
         54.99,  # Northern boundary (approximate)
         230,  # Western boundary (approximate)
         299.99  # Eastern boundary (approximate)
)

SA_REGION = (
      -37.75,  # Southern boundary (approximate)
         -22.00,  # Northern boundary (approximate)
         15.25,   # Western boundary (approximate)
         35.00,  # Eastern boundary (approximate)
)

EU_REGION = (
      35.00,  # Southern boundary (approximate)
         69.9,  # Northern boundary (approximate)
         -10.00,  # Western boundary (approximate)
         39.9  # Eastern boundary (approximate)
)



results = evaluation_between_regions(
                        model,
                        era5_data=sliced_era5, 
                        hres_data=sliced_hrest0,
                        target_region=SA_REGION,
                        base_region= USA_REGION,
                        third_region= EU_REGION,
                        device=device
               )



eu_region_surface_rmses = results['third_region_surface_rmses']
eu_region_atmospheric_rmses = results['third_region_atmospheric_rmses']

target_region_surface_rmses = results['target_region_surface_rmses']
target_region_atmospheric_rmses = results['target_region_atmospheric_rmses']
base_region_surface_rmses = results['base_region_surface_rmses']
base_region_atmospheric_rmses = results['base_region_atmospheric_rmses']



lead_time = [6, 12, 18, 24, 30, 36, 42, 48]
SELECTED_ATMOS_LEVELS1 = {1:"100 hPa", 2:"150 hPa", 3: "200 hPa", 
                         4:"250 hPa", 5:"300 hPa"}
SELECTED_ATMOS_LEVELS2 = {6:"400 hPa",8: "600 hPa", 9: "700 hPa", 10:"850 hPa", 11: "925 hPa"}



def plot(SELECTED_ATMOS_LEVELS, level=1):
    target_region_surface_atmos_rmses={}
    base_region_surface_atmos_rmses={}
    eu_region_surface_atmos_rmses={}

    for num, name in SELECTED_ATMOS_LEVELS.items():
        for var, rmses in target_region_atmospheric_rmses.items():
            target_region_surface_atmos_rmses[f"{var} {name}"] = target_region_atmospheric_rmses[var][num]
            
        
    for num, name in SELECTED_ATMOS_LEVELS.items():
        for var, rmses in base_region_atmospheric_rmses.items():
            base_region_surface_atmos_rmses[f"{var} {name}"] = base_region_atmospheric_rmses[var][num]
            


    for num, name in SELECTED_ATMOS_LEVELS.items():
        for var, rmses in eu_region_atmospheric_rmses.items():
            eu_region_surface_atmos_rmses[f"{var} {name}"] = eu_region_atmospheric_rmses[var][num]
            
        
            
    num_plots = len(target_region_surface_atmos_rmses)

    num_plots = len(target_region_surface_atmos_rmses)
    num_plots_per_rows = 5
    num_rows = int(np.ceil(num_plots/num_plots_per_rows)) ##  to check
    variables = list(target_region_surface_atmos_rmses.keys())






    saving_path = "../report/evaluation/rmses_grid/fine_tuned_small"

    fig, axs = plt.subplots(num_rows, num_plots_per_rows, dpi=300, figsize=(20, 20))
    axs = axs.ravel()

    # Store handles and labels from the first plot for global legend
    handles, labels = None, None

    for i, ax in enumerate(axs[:num_plots]):
        line1, = ax.plot(lead_time, target_region_surface_atmos_rmses[variables[i]], label="South Africa", c="brown")
        line3, = ax.plot(lead_time, eu_region_surface_atmos_rmses[variables[i]], label="Europe", c="navy")
        
        line2, = ax.plot(lead_time, base_region_surface_atmos_rmses[variables[i]], label="USA", c="teal")
        ax.set_title(variables[i])
        ax.grid(True)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Turn off unused axes
    for ax in axs[num_plots:]:
        ax.axis('off')

    # Add shared labels
    fig.supxlabel("Lead Time (Hours)")
    fig.supylabel("RMSE")

    # Add a single global legend below x-axis label
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout(pad=1.3)
    plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu_wampln_sup_atmos{level}.pdf", bbox_inches="tight")
    plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu_wampln_sup_atmos{level}.png", bbox_inches="tight")
    plt.savefig(f"{saving_path}/sa_vs_usa_vs_eu_wampln_sup_atmos{level}.svg", bbox_inches="tight")


plot(SELECTED_ATMOS_LEVELS1, level=1)
plot(SELECTED_ATMOS_LEVELS2, level=2)
