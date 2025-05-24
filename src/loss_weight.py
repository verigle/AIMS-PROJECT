import torch
import torch.nn as nn

# Example usage:
# Define weights
ALPHA = 1/4
BETA = 1.0
DATASET_WEIGHTS = {'ERA5': 2.0, 'GFS-T0': 1.5, "OTHER": 1}  # Dataset-specific weights
SURFACE_WEIGHTS = {
    "2t": 3,    # Surface temperature, keep it high
    "10u": 0.77,
    "10v": 0.66,
    "msl": 3
}

ATMOSPHERIC_WEIGHTS = {
    "z": 0.5,   # Geopotential, lower priority
    "q": 5.0,   # Specific humidity, **higher priority**
    "t": 4.0,   # Atmospheric temperature, **higher priority**
    "u": 1.0,   # Zonal wind, slightly less
    "v": 1.2    # Meridional wind, slightly less
}



class AuroraLoss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA, dataset_weights=DATASET_WEIGHTS,
                 surface_weights=SURFACE_WEIGHTS, 
                 atmospheric_weights=ATMOSPHERIC_WEIGHTS,
                 num_surface_var=4,
                 num_atmos_var=5,
                 num_atmos_levels=13):
        super(AuroraLoss, self).__init__()
        self.alpha = alpha  # Weight for surface loss
        self.beta = beta    # Weight for atmospheric loss
        self.dataset_weights = dataset_weights  # Dataset-specific weights
        self.surface_weights = surface_weights  # Weights for surface variables
        self.atmospheric_weights = atmospheric_weights  # Weights for atmospheric variables
        self.num_surface_vars = num_surface_var
        self.num_atmos_vars = num_atmos_var
        self.num_atmos_levels = num_atmos_levels

    def forward(self, pred, target, dataset_name):
        # Retrieve dataset-specific weight
        gamma = self.dataset_weights.get(dataset_name, 1.0)

        # Surface variables loss
        # VS =  self.num_surface_vars
        surface_loss = 0
        for surf_var, weight in self.surface_weights.items():
            pred_tensor = pred.surf_vars[surf_var].squeeze()
            pred_tensor = pred_tensor.to("cuda")
            target_tensor = target.surf_vars[surf_var].squeeze()[0,:,:]
            target_tensor = target_tensor.to("cuda")
            
            abs_diff = torch.abs(pred_tensor - target_tensor)
            
            H, W = target_tensor.shape[-2:]
            
            
            weighted_diff = weight * abs_diff.sum()/(W*H)
            
            
            surface_loss += weighted_diff 
        surface_loss = surface_loss*self.alpha

        # Atmospheric variables loss
        C =  self.num_atmos_levels
        atmos_loss = 0
        for atmos_var, weight in self.atmospheric_weights.items():
            var_loss = 0
        
            for c in range(C):
                
                pred_tensor= pred.atmos_vars[atmos_var].squeeze()[c,:,:]
                pred_tensor = pred_tensor.to("cuda")
                target_tensor = target.atmos_vars[atmos_var].squeeze()[0,c,:,:]
                
                target_tensor = target_tensor.to("cuda")
                
                abs_diff = torch.abs(pred_tensor - target_tensor)
                
                H, W = target_tensor.shape[-2:]
                
                
                weighted_diff = weight * abs_diff.sum()
                
                
                var_loss += weighted_diff 
            atmos_loss += var_loss/(H*W*C)
        atmos_loss = atmos_loss*self.beta
        

            
           
        # Total loss with dataset-specific weighting
        total_loss = gamma/(self.num_surface_vars+self.num_atmos_vars) * (surface_loss + atmos_loss)
        return total_loss

# Initialize the loss function
# criterion = AuroraLoss(alpha, beta, dataset_weights, surface_weights, atmospheric_weights)


