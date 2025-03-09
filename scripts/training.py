#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import sys
sys.path.append(os.path.abspath("../src"))
from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses, custom_rmse



def print_trainable_parameters(model):
    parameters, trainable = 0, 0
    
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(f"trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)")



model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)
model.load_state_dict(torch.load('../model/aurora-pretrained.pth'))





for param in model.parameters():
    param.requires_grad = False



# In[8]:


from functools import partial
from lora import LinearWithLoRA
# default hyperparameter choices
lora_r = 8
lora_alpha = 16
# lora_dropout = 0.05
# lora_query = True
# lora_key = False
# lora_value = True
# lora_projection = False
# lora_mlp = False
# lora_head = False

# layers = []

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)


# # Add lora to some parts

# # Backbone

# ##  MLP

# In[9]:


model.backbone.time_mlp[0] = assign_lora(model.backbone.time_mlp[0])
model.backbone.time_mlp[2] = assign_lora(model.backbone.time_mlp[2])


# ## Encoder

# In[10]:


for block in model.backbone.encoder_layers:
    for layer in block.blocks:
        layer.norm1.ln_modulation[1] = assign_lora(layer.norm1.ln_modulation[1])
        layer.attn.qkv = assign_lora(layer.attn.qkv)
        layer.attn.proj = assign_lora(layer.attn.proj)
        layer.norm2.ln_modulation[1] =  assign_lora(layer.norm2.ln_modulation[1])
        layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
        layer.mlp.fc2 = assign_lora(layer.mlp.fc2)
    if  block.downsample:
        block.downsample.reduction = assign_lora(block.downsample.reduction)

    


# ## Decoder

# In[11]:


for block in model.backbone.decoder_layers:
    for layer in block.blocks:
        layer.norm1.ln_modulation[1] = assign_lora(layer.norm1.ln_modulation[1])
        layer.attn.qkv = assign_lora(layer.attn.qkv)
        layer.attn.proj = assign_lora(layer.attn.proj)
        layer.norm2.ln_modulation[1] =  assign_lora(layer.norm2.ln_modulation[1])
        layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
        layer.mlp.fc2 = assign_lora(layer.mlp.fc2)
    if  block.upsample:
        block.upsample.lin1 = assign_lora(block.upsample.lin1)
        block.upsample.lin2 = assign_lora(block.upsample.lin2)

    


# In[12]:


print_trainable_parameters(model)


# # Get south africa Data

# In[13]:


fs = gcsfs.GCSFileSystem(token="anon")

store = fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
full_era5 = xr.open_zarr(store=store, consolidated=True, chunks=None)


# In[14]:


# start_time, end_time = '2022-11-01', '2023-01-31'
start_time, end_time = '2022-12-01', '2023-01-31'

# In[15]:


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



from train import training
import torch.optim as optim


# In[17]:


optimizer = optim.Adam(model.parameters(), lr=1e-4)


# In[18]:


from loss import AuroraLoss


# In[19]:


criterion = AuroraLoss()


# In[ ]:


model,  rmses = training(model=model, criterion=criterion,
             num_epochs=20, optimizer=optimizer,
             dataset= sliced_era5_SA,
             dataset_name="ERA5", 
             accumulation_steps=8,
             checkpoint_dir = '../model/checkpoints')


torch.save(model.state_dict(), "../model/best_models/best_model.pth")
save_path  = "../report/training"
_, ax = plt.subplots(figsize=(8, 6), dpi=300)

ax.plot(np.arange(1,len(rmses)+1), rmses)
ax.set_ylabel("Mean Absolute Error (MAP)")
ax.set_xlabel("Epoch")
plt.savefig(f"{save_path}/map-learning-curve.pdf", bbox_inches="tight")
plt.savefig(f"{save_path}/map-learning-curve.png", bbox_inches="tight")
plt.savefig(f"{save_path}/map-learning-curve.svg", bbox_inches="tight")

