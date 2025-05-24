import torch
import torch.nn as nn

# Define weights
ALPHA = 1/4
BETA = 1.0
DATASET_WEIGHTS = {'ERA5': 2.0, 'GFS-T0': 1.5, "OTHER": 1}
SURFACE_WEIGHTS = {
    "2t": 1.5,  # Smaller weights now
    "10u": 1.0,
    "10v": 1.0,
    "msl": 1.5
}
ATMOSPHERIC_WEIGHTS = {
    "z": 1.0,
    "q": 2.0,  # Boost q a little bit
    "t": 1.5,  # Boost t a little bit
    "u": 1.0,
    "v": 1.0
}

class AuroraLoss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA, dataset_weights=DATASET_WEIGHTS,
                 surface_weights=SURFACE_WEIGHTS,
                 atmospheric_weights=ATMOSPHERIC_WEIGHTS,
                 num_surface_var=4, num_atmos_var=5, num_atmos_levels=13,
                 use_relative_error=True, use_weights=True, epsilon=1e-6):
        super(AuroraLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dataset_weights = dataset_weights
        self.surface_weights = surface_weights
        self.atmospheric_weights = atmospheric_weights
        self.num_surface_vars = num_surface_var
        self.num_atmos_vars = num_atmos_var
        self.num_atmos_levels = num_atmos_levels
        self.use_relative_error = use_relative_error
        self.use_weights = use_weights
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def compute_error(self, pred, target):
        if self.use_relative_error:
            return torch.abs(pred - target) / (torch.abs(target) + self.epsilon)
        else:
            return torch.abs(pred - target)

    def forward(self, pred, target, dataset_name):
        # Retrieve dataset-specific weight
        gamma = self.dataset_weights.get(dataset_name, 1.0)

        # Surface loss
        surface_loss = 0
        for surf_var in self.surface_weights.keys():
            pred_tensor = pred.surf_vars[surf_var].squeeze()
            target_tensor = target.surf_vars[surf_var].squeeze()[0, :, :]

            error = self.compute_error(pred_tensor, target_tensor)

            H, W = target_tensor.shape[-2:]
            weight = self.surface_weights[surf_var] if self.use_weights else 1.0
            weighted_diff = weight * error.sum() / (W * H)
            surface_loss += weighted_diff

        surface_loss = surface_loss * self.alpha

        # Atmospheric loss
        atmos_loss = 0
        C = self.num_atmos_levels
        for atmos_var in self.atmospheric_weights.keys():
            var_loss = 0
            for c in range(C):
                pred_tensor = pred.atmos_vars[atmos_var].squeeze()[c, :, :]
                target_tensor = target.atmos_vars[atmos_var].squeeze()[0, c, :, :]

                error = self.compute_error(pred_tensor, target_tensor)

                H, W = target_tensor.shape[-2:]
                weight = self.atmospheric_weights[atmos_var] if self.use_weights else 1.0
                weighted_diff = weight * error.sum()
                var_loss += weighted_diff

            atmos_loss += var_loss / (H * W * C)

        atmos_loss = atmos_loss * self.beta

        # Total loss
        total_loss = gamma / (self.num_surface_vars + self.num_atmos_vars) * (surface_loss + atmos_loss)
        return total_loss
