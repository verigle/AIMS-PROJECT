import torch
import numpy as np

def evaluation_rmse_weights(latitudes: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Compute latitude-based weighting factors for RMSE evaluation.
    
    The weights are computed based on the cosine of the latitude values,
    normalized such that their sum equals the number of latitude points.
    
    Parameters:
    - latitudes (torch.Tensor): Tensor containing latitude values in degrees.
    - device (str): Device to perform computation (default is "cuda").
    
    Returns:
    - torch.Tensor: Normalized latitude-based weights.
    """
    latitudes = torch.deg2rad(latitudes.to(device))  # Convert degrees to radians
    weights = torch.cos(latitudes)
    weights = weights * len(latitudes) / torch.sum(weights)
    return weights

def evaluation_rmse(
    actual: torch.Tensor, 
    prediction: torch.Tensor, 
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute the root mean square error (RMSE) with latitude-based weighting.
    
    The RMSE is computed over a 2D spatial grid, where latitude-dependent weights
    are applied to account for the varying grid cell areas.
    
    Parameters:
    - actual (torch.Tensor): Ground truth values (H x W).
    - prediction (torch.Tensor): Predicted values (H x W).
    - latitudes (torch.Tensor): Latitude values in degrees (H, ).
    - longitudes (torch.Tensor): Longitude values (W, ).
    - device (str): Device to perform computation (default is "cuda").
    
    Returns:
    - torch.Tensor: Weighted RMSE value.
    """
    actual, prediction = actual.to(device), prediction.to(device)
    latitudes_weights = evaluation_rmse_weights(latitudes, device)
    
    H, W = actual.shape
    squared_error = (actual - prediction) ** 2
    
    # Create a 2D grid of weights
    area_grid_weights = torch.outer(latitudes_weights, torch.ones(len(longitudes), device=device))
    
    # Compute weighted RMSE
    rmse = torch.mean(torch.sqrt(area_grid_weights * squared_error/(H*W)))
    return rmse
