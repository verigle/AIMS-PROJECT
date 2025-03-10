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
from utils import rmse_fn, plot_rmses, predict_train_fn


import os
import torch
from utils import (
    get_surface_feature_target_data,
    get_atmos_feature_target_data,
    get_static_feature_target_data,
    create_batch
)
import logging
import sys

# Configure logging
log_file = "trainings.log"
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console as well
    ]
)

logger = logging.getLogger(__name__)


def training(model, criterion, num_epochs,
             optimizer, dataset=None, dataset_name="ERA5", 
             accumulation_steps=8,
             rollouts_num=8,
             checkpoint_dir='../model/checkpoints'):
    selected_times = dataset.time
    loss_list = []

    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Initialize gradients
        running_loss = 0
        for i in range(0, len(selected_times) - rollouts_num-1):
            # Retrieve data for current and next time steps
            sa_feature_data = dataset.sel(time=slice(selected_times[i], selected_times[i+1]))
            sa_target_data = dataset.sel(time=slice(selected_times[i + rollouts_num], selected_times[i + rollouts_num + 1]))
            # sa_target_data = dataset.sel(time=selected_times[i + rollouts_num + 1])

            # Extract feature and target data
            sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_data, sa_target_data)
            sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_data, sa_target_data)
            sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_data, sa_target_data)

            # Create input and target batches
            input_batch = create_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data)
            target_batch = create_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data)

            # Forward pass
            outputs = predict_train_fn(model=model, batch=input_batch)
            prediction_48h = outputs[-1]
            # output = model(input_batch)
            loss = criterion(prediction_48h, target_batch, dataset_name) / accumulation_steps  # Normalize loss
            print(loss.item())
            
             # Accumulate loss
            running_loss += loss.item() 

            # Backward pass
            loss.backward()

            # Update weights and reset gradients every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(selected_times) - 3:
                optimizer.step()
                optimizer.zero_grad()
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(selected_times)
        # Record loss
        loss_list.append(epoch_loss)
 
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model, loss_list


    

    
