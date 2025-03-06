import torch
import torch.nn as nn

# Example usage:
# Define weights
ALPHA = 1/4
BETA = 1.0
DATASET_WEIGHTS = {'ERA5': 2.0, 'GFS-T0': 1.5}  # Dataset-specific weights
SURFACE_WEIGHTS = {"2t":3.0, "10u":0.77, "10v":0.66, "msl":1.5} # Weights for MSL, U10, V10, 2T
ATMOSPHERIC_WEIGHTS = {"z":2.8, "q":0.78, "t":1.7, "u":0.87, "v":0.6}



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
            pred_tensor = pred.surf_vars[surf_var][0, 0]
            target_tensor = target.surf_vars[surf_var].squeeze()
            
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
                
                pred_tensor= pred.atmos_vars[atmos_var].squeeze()[c,:,:].squeeze()
                target_tensor = target.atmos_vars[atmos_var].squeeze()[:,c,:,:]
                
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


