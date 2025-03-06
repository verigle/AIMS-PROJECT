import xarray as xr
from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout
import matplotlib.pyplot as plt


from torch.utils.data import Dataset
from aurora import Batch, Metadata
import os
import torch.optim as optim
from loss import AuroraLoss

from utils import get_surface_feature_target_data, get_atmos_feature_target_data
from utils import get_static_feature_target_data, create_batch, predict_fn, rmse_weights
from utils import rmse_fn, plot_rmses


def training(model, criterion,
             num_epochs, optimizer,
             dataset=None,
             dataset_name="ERA5", 
             accumulation_steps=8
             ):
    selected_times = dataset.time
    loss_list=[]
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(selected_times)-3):
            # get current and previous time step data

            sa_feature_data =  (
                    dataset
                    .sel(time=slice(selected_times[i], selected_times[i+1]))
                )

            sa_target_data =  (
                    dataset
                    .sel(time=slice(selected_times[i+2], selected_times[i+3]))
                )
            
            # get each type of data(surface, static atmosphere)

            sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_data, sa_target_data)
            sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_data, sa_target_data)
            sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_data, sa_target_data)
            
            # create batch for each of them

            input =  create_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
            target = create_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)
            
                        
            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, target, dataset_name)
            loss = loss / accumulation_steps  # Normalize loss
            print(loss.detach().numpy())
            
            # Backward pass
            loss.backward()
            
            # Update weights and reset gradients every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining gradients if dataset size is not divisible by accumulation_steps
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        loss_list.append(loss.detach().numpy())
        
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    return model, loss_list


    

    
