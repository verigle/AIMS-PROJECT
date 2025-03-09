from torch import nn
import torch
from functools import partial
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    

def create_custom_model(model, lora_r = 8, lora_alpha = 16):
    for param in model.parameters():
        param.requires_grad = False
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    model.backbone.time_mlp[0] = assign_lora(model.backbone.time_mlp[0])
    model.backbone.time_mlp[2] = assign_lora(model.backbone.time_mlp[2])
    
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

    
    return model
    