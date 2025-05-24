import xarray as xr
import torch
import os
import logging
import sys
import torch.optim as optim
from datetime import datetime

from aurora import AuroraSmall, Batch, Metadata, rollout
from torch.utils.data import Dataset
from loss import AuroraLoss
from utils import (
    get_surface_feature_target_data,
    get_atmos_feature_target_data,
    get_static_feature_target_data,
    create_batch,
    create_hrest0_batch,
    predict_train_fn
)

# Configure logging
log_file = "training_model3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def training(model, criterion, num_epochs, optimizer, era5_data=None, hres_data=None,
             dataset_name="HRES", accumulation_steps=32, rollouts_num=8, checkpoint_dir='../model/training/hrest0/model3'):
    
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    
    selected_times = hres_data.time.values  # Extract time values as NumPy array (faster access)
    num_samples = len(selected_times) - rollouts_num - 2
    loss_list = []

    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU if available

    # model.configure_activation_checkpointing()
    model.train()
    
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Initialize gradients
        running_loss = 0

        for i in range(num_samples):
            t0, t1, t2, t3 = selected_times[i], selected_times[i+1], selected_times[i+1+rollouts_num], selected_times[i+2+rollouts_num]

            # Load required time slices once (avoids redundant slicing)
            sa_feature_hrest0_data = hres_data.sel(time=slice(t0, t1))
            sa_target_hrest0_data = hres_data.sel(time=slice(t2, t3))
            sa_feature_era5_data = era5_data.sel(time=slice(t0, t1))
            sa_target_era5_data = era5_data.sel(time=slice(t2, t3))

            # Extract features and targets
            sa_feature_surface_data, sa_target_surface_data = get_surface_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
            sa_feature_atmos_data, sa_target_atmos_data = get_atmos_feature_target_data(sa_feature_hrest0_data, sa_target_hrest0_data)
            sa_feature_static_data, sa_target_static_data = get_static_feature_target_data(sa_feature_era5_data, sa_target_era5_data)

            # Create input and target batches
            input_batch = create_hrest0_batch(sa_feature_surface_data, sa_feature_atmos_data, sa_feature_static_data).to(device)
            target_batch = create_hrest0_batch(sa_target_surface_data, sa_target_atmos_data, sa_target_static_data).to(device)

            # Forward pass
            outputs = predict_train_fn(model=model, batch=input_batch)
            prediction_48h = outputs[-1]

            # Compute loss and accumulate
            loss = criterion(prediction_48h, target_batch, dataset_name)/ accumulation_steps
            # Backward pass
            loss.backward()
            
            running_loss += loss.item()  
            
            # Check for NaNs in loss
            if torch.isnan(loss).any():
                logger.info(f"NaN detected in loss at iteration {i}")

          

            # Update weights and reset gradients every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_samples:
                optimizer.step()
                optimizer.zero_grad()

        # Calculate epoch loss
        # epoch_loss = running_loss / num_samples
        epoch_loss = running_loss / (num_samples / accumulation_steps)

        loss_list.append(epoch_loss)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss}, checkpoint_path)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        logger.info(f'Checkpoint saved: {checkpoint_path}')

    return model, loss_list
