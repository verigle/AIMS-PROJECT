from torch import nn
import torch
from functools import partial
import torch
import math
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, 
                 out_dim, rank, 
                 alpha, dropout: float = 0.0):
        super().__init__()
        self.scaling = alpha / rank
        
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Initialise A the same way as the default for `nn.Linear` and set B to zero.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        x = self.scaling * (self.lora_dropout(x) @ self.A @ self.B)
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
    

def create_custom_model(model, lora_r = 8, lora_alpha = 1):
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
    
    
def full_linear_layer_lora(model, lora_r = 8, lora_alpha = 1):
    for param in model.parameters():
        param.requires_grad = False
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    
    # Encoder
    model.encoder.surf_mlp.net[0] = assign_lora(model.encoder.surf_mlp.net[0])
    model.encoder.surf_mlp.net[2] = assign_lora(model.encoder.surf_mlp.net[2]) 
    model.encoder.pos_embed = assign_lora(model.encoder.pos_embed) 
    model.encoder.scale_embed = assign_lora(model.encoder.scale_embed) 
    model.encoder.absolute_time_embed = assign_lora(model.encoder.absolute_time_embed) 
    model.encoder.atmos_levels_embed = assign_lora(model.encoder.atmos_levels_embed) 
    model.encoder.level_agg.layers[0][0].to_q = assign_lora(model.encoder.level_agg.layers[0][0].to_q) 
    model.encoder.level_agg.layers[0][0].to_kv = assign_lora(model.encoder.level_agg.layers[0][0].to_kv)
    model.encoder.level_agg.layers[0][0].to_out = assign_lora(model.encoder.level_agg.layers[0][0].to_out)
    model.encoder.level_agg.layers[0][1].net[0] = assign_lora(model.encoder.level_agg.layers[0][1].net[0])
    model.encoder.level_agg.layers[0][1].net[2] = assign_lora(model.encoder.level_agg.layers[0][1].net[2])
    
    # Backbone
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
    
    # decoder
    model.decoder.level_decoder.layers[0][0].to_q = assign_lora(model.decoder.level_decoder.layers[0][0].to_q) 
    model.decoder.level_decoder.layers[0][0].to_kv = assign_lora(model.decoder.level_decoder.layers[0][0].to_kv)
    model.decoder.level_decoder.layers[0][0].to_out = assign_lora(model.decoder.level_decoder.layers[0][0].to_out)
    model.decoder.level_decoder.layers[0][1].net[0] = assign_lora(model.decoder.level_decoder.layers[0][1].net[0])
    model.decoder.level_decoder.layers[0][1].net[2] = assign_lora(model.decoder.level_decoder.layers[0][1].net[2])
    model.decoder.surf_heads["10u"] = assign_lora(model.decoder.surf_heads["10u"])
    model.decoder.surf_heads["10v"] = assign_lora(model.decoder.surf_heads["10v"])
    model.decoder.surf_heads["2t"] = assign_lora(model.decoder.surf_heads["2t"])
    model.decoder.surf_heads["msl"] = assign_lora(model.decoder.surf_heads["msl"])
    model.decoder.atmos_heads["q"] = assign_lora(model.decoder.atmos_heads["q"])
    model.decoder.atmos_heads["t"] = assign_lora(model.decoder.atmos_heads["t"])
    model.decoder.atmos_heads["u"] = assign_lora(model.decoder.atmos_heads["u"])
    model.decoder.atmos_heads["v"] = assign_lora(model.decoder.atmos_heads["v"])
    model.decoder.atmos_heads["z"] = assign_lora(model.decoder.atmos_heads["z"])
    
    return model
    

    
def print_trainable_parameters(model):
    parameters, trainable = 0, 0
    
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return f"trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)"
