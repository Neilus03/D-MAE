import torch
import torch.nn as nn
import numpy as np
import sys, os, yaml
import matplotlib.pyplot as plt
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from model.vit import ViTFeatureExtractor, PatchEmbedding, TransformerEncoder, PositionalEncoding
from data.dataloader import denormalize_RGB


config_path = '/home/ndelafuente/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
wandb.init(project=config['logging']['wandb_project'], entity=config['logging']['wandb_entity'])

class RandomMasking(nn.Module):
    '''
    Random Masking class:
    This class is used to randomly mask patches from the input
    '''
    def __init__(self, mask_ratio=config['training']['mask_ratio']):
        '''
        __init__ function:
        This function is used to initialize the RandomMasking class
        Args:
            mask_ratio: Ratio of patches to be masked
        '''
        super().__init__()
        self.mask_ratio = mask_ratio
        print(f"Initialized RandomMasking with mask_ratio: {self.mask_ratio}")

    def forward(self, x):
        '''
        forward function:
        This function is used to randomly mask patches
        Args:
            x: Input patches (B, L, D)
        Returns:
            x_masked: Masked patches (B, L_visible, D)
            mask: Binary mask (B, L)
            ids_restore: Indices to restore original order (B, L)
        '''
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        # Generate random mask
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Define ids_keep: Indices of patches to keep
        ids_keep = ids_shuffle[:, :len_keep] 

        # Create binary mask: 1 is keep, 0 is remove
        mask = torch.zeros(N, L, device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)

        print(f"RandomMasking - Input patches shape: {x.shape}")
        print(f"RandomMasking - Mask shape: {mask.shape}")
        print(f"RandomMasking - ids_restore shape: {ids_restore.shape}")

        # Apply mask to input
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # Gather the visible patches
        print(f"RandomMasking -  Unmasked patches shape: {x_masked.shape}")

        return x_masked, mask, ids_restore

def unpatchify(patches, img_size=(224,224), patch_size=16, original_num_patches=196, save_dir='./unpatchify_debug'):
    '''
    Reconstructs the image from patches.
    
    Args:
        patches (torch.Tensor): The output patches from the decoder
                                (shape: [batch_size, num_patches, patch_size*patch_size*num_channels]).
        img_size (tuple): The original image size (height, width).
        patch_size (int): The size of each patch (assuming square patches).
        
    Returns:
        torch.Tensor: The reconstructed image (shape: [batch_size, 3, height, width]).
        torch.Tensor: The reconstructed depth map (shape: [batch_size, 1, height, width]).

    '''
    batch_size, num_patches, patch_elements = patches.shape
    num_channels_rgb = 3
    num_channels_depth = 1

    # Calculate the number of patches in each dimension for the original image
    num_patches_height = img_size[0] // patch_size
    num_patches_width = img_size[1] // patch_size

    # Calculate the elements per patch for RGB and depth
    patch_elements_rgb = patch_size * patch_size * num_channels_rgb
    patch_elements_depth = patch_size * patch_size * num_channels_depth

    # Separate RGB and depth patches
    patches_rgb = patches[:, :, :patch_elements_rgb]
    patches_depth = patches[:, :, patch_elements_rgb:patch_elements_rgb + patch_elements_depth]

    print(f"Unpatchify - RGB patches shape before reshaping: {patches_rgb.shape}")
    print(f"Unpatchify - Depth patches shape before reshaping: {patches_depth.shape}")

    # Visualize RGB patches for the first image in the batch
    visualize_patches(patches_rgb[0], patch_size, "RGB", save_dir=save_dir, step=0)

    # Reshape patches to match the original image dimensions
    patches_rgb = patches_rgb.reshape(batch_size, num_patches, patch_size, patch_size, num_channels_rgb)
    patches_rgb = patches_rgb.permute(0, 1, 3, 2, 4).contiguous()
    
    # Create a tensor of zeros for the full image and fill in the visible patches
    full_patches_rgb = torch.zeros(batch_size, original_num_patches, patch_size, patch_size, num_channels_rgb, device=patches_rgb.device)
    full_patches_rgb[:, :num_patches] = patches_rgb
    
    # Reshape to the original image dimensions
    reconstructed_rgb = full_patches_rgb.reshape(batch_size, num_patches_height, num_patches_width, patch_size, patch_size, num_channels_rgb)
    reconstructed_rgb = reconstructed_rgb.permute(0, 5, 1, 3, 2, 4).contiguous()
    reconstructed_rgb = reconstructed_rgb.reshape(batch_size, num_channels_rgb, img_size[0], img_size[1])

    # Repeat the process for depth patches
    if patches_depth.shape[2] > 0:
        patches_depth = patches_depth.reshape(batch_size, num_patches, patch_size, patch_size, num_channels_depth)
        patches_depth = patches_depth.permute(0, 1, 3, 2, 4).contiguous()
        
        full_patches_depth = torch.zeros(batch_size, original_num_patches, patch_size, patch_size, num_channels_depth, device=patches_depth.device)
        full_patches_depth[:, :num_patches] = patches_depth
        
        reconstructed_depth = full_patches_depth.reshape(batch_size, num_patches_height, num_patches_width, patch_size, patch_size, num_channels_depth)
        reconstructed_depth = reconstructed_depth.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed_depth = reconstructed_depth.reshape(batch_size, num_channels_depth, img_size[0], img_size[1])
    else:
        reconstructed_depth = torch.zeros(batch_size, num_channels_depth, img_size[0], img_size[1], device=patches_rgb.device)

    print(f"Unpatchify - RGB image shape after reshaping: {reconstructed_rgb.shape}")
    print(f"Unpatchify - Depth image shape after reshaping: {reconstructed_depth.shape}")

    return reconstructed_rgb, reconstructed_depth


def visualize_patches(patches, patch_size, label, save_dir='./unpatchify_debug', step=0):
    """
    Visualize the individual patches for debugging.
    Args:
        patches (torch.Tensor): Tensor of patches.
        patch_size (int): Size of each patch.
        label (str): Label for the plot.
        save_dir (str): Directory to save the visualization.
        step (int): Current step/epoch for naming files.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_patches = patches.shape[0]
    grid_size = int(num_patches ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            # Detach the tensor from the graph and convert it to numpy
            patch = patches[i * grid_size + j].detach().view(patch_size, patch_size, -1).cpu().numpy()
            axes[i, j].imshow(patch)
            axes[i, j].axis('off')
    plt.suptitle(f"{label} Patches")
    filename = os.path.join(save_dir, f"{label}_patches_step_{step}.png")
    plt.savefig(filename)
    plt.close()

    # Log the image to wandb
    wandb.log({f"{label} Patches at Step {step}": wandb.Image(filename)})


class MAEEncoder(nn.Module):
    '''
    MAE Encoder class:
    This class is used to encode the visible patches
    '''
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers):
        '''
        __init__ function:
        This function is used to initialize the MAEEncoder class
        Args:
            d_model: Model dimension
            img_size: Image size
            patch_size: Patch size
            n_channels: Number of channels
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        '''
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, (img_size[0] // patch_size) * (img_size[1] // patch_size))
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads, n_layers) for _ in range(n_layers)])

    def forward(self, x):
        '''
        forward function:
        This function is used to encode the visible patches
        Args:
            x: Input visible patches (B, C, H, W)
        Returns:
            x: Encoded patches (B, L, D)
        '''
        print(f"MAEEncoder - Input patches shape: {x.shape}")
        
        x = self.transformer_encoder(x)
        print(f"MAEEncoder - After transformer encoder: {x.shape}")
        return x

class MAEDecoder(nn.Module):
    '''
    MAE Decoder class:
    This class is used to decode the encoded patches and reconstruct the full image
    '''
    def __init__(self, d_model, num_patches, patch_size, num_channels=4, n_layers=8, n_heads=4):
        '''
        __init__ function:
        This function is used to initialize the MAEDecoder class
        Args:
            d_model: Model dimension
            num_patches: Number of patches
            patch_size: Patch size
            num_channels: Number of channels (4 for RGB + depth)
            n_layers: Number of transformer layers
        '''
        super().__init__()
        self.decoder_embed = nn.Linear(d_model, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoder(d_model, n_heads=n_heads, n_layers=n_layers)
            for _ in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_pred = nn.Linear(d_model, patch_size * patch_size * num_channels)
    
    def get_decoder_pred(self):
        '''
        get_decoder_pred function accessible from outside the class
        Returns:
            self.decoder_pred: Linear layer for prediction
        '''
        return self.decoder_pred

    def forward(self, x, ids_restore):
        '''
        forward function:
        This function is used to decode the encoded patches and reconstruct the full image
        Args:
            x: Encoded patches (B, L_visible, D)
            ids_restore: Indices to restore original order (B, L)
        Returns:
            x: Reconstructed patches (B, L, patch_size*patch_size*num_channels)
        '''
        print(f"MAEDecoder - Input encoded patches shape: {x.shape}")
        print(f"MAEDecoder - ids_restore shape: {ids_restore.shape}")
        x = self.decoder_embed(x)
        print(f"MAEDecoder - After embedding: {x.shape}")
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        print(f"MAEDecoder - Mask tokens shape: {mask_tokens.shape}")
        x_ = torch.cat([x, mask_tokens], dim=1)
        print(f"MAEDecoder - Concatenated shape: {x_.shape}")
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        print(f"MAEDecoder - Gathered shape: {x_.shape}")

        for block in self.decoder_blocks:
            x_ = block(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        print(f"MAEDecoder - Output (predicted patches) shape: {x_.shape}")

        return x_

class MAE(nn.Module):
    '''
    MAE class:
    This class implements the full Masked Autoencoder
    '''
    def __init__(self, d_model=config['model']['d_model'],
                 img_size=config['model']['image_size'],
                 patch_size=config['model']['patch_size'],
                 n_channels=config['model']['n_channels'],
                 n_heads_encoder=config['model']['num_heads_encoder'],
                 n_layers_encoder=config['model']['num_layers_encoder'],
                 n_heads_decoder=config['model']['num_heads_encoder'],
                 n_layers_decoder=config['model']['num_layers_encoder'],
                 mask_ratio=config['training']['mask_ratio']):
        '''
        __init__ function:
        This function is used to initialize the MAE class
        Args:
            d_model: Model dimension
            img_size: Image size
            patch_size: Patch size
            n_channels: Number of channels (4 for RGB + depth)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            mask_ratio: Ratio of patches to be masked
        '''
        super().__init__()
        self.random_masking = RandomMasking(mask_ratio)
        self.encoder = MAEEncoder(d_model, img_size, patch_size, n_channels, n_heads_encoder, n_layers_encoder)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) # for 224x224 image and 16x16 patches, num_patches = 196, 49 visible and 147 masked
        self.decoder = MAEDecoder(d_model, num_patches, patch_size, n_channels, n_layers_decoder, n_heads_decoder)
    
    def get_decoder_pred(self):
        return self.decoder.get_decoder_pred()

    def forward(self, x):
        '''
        forward function:
        This function implements the forward pass of the MAE
        Args:
            x: Input image (B, C, H, W)
        Returns:
            pred: Reconstructed patches (B, L, patch_size*patch_size*num_channels)
            mask: Binary mask (B, L) - 0 is keep, 1 is remove
            ids_restore: Indices to restore original order (B, L)
        '''
        
        print(f"MAE - Input image shape: {x.shape}")
        # Apply patch embedding
        x_patched = self.encoder.patch_embed(x)
        print(f"MAE - After patch embedding: {x_patched.shape}")
        
        # Apply positional encoding BEFORE masking
        x = self.encoder.positional_encoding(x_patched)
        print(f"MAE - After positional encoding: {x.shape}")
        
        # Apply masking to the patches
        x, mask, ids_restore = self.random_masking(x)
        
        # Encode the masked patches
        x_encoded = self.encoder(x)
        
        # Decode patches
        pred = self.decoder(x_encoded, ids_restore)
        print(f"MAE - Reconstructed patches shape: {pred.shape}")
        
        # Unpatchify the patches to reconstruct the image and depth map
        reconstructed_image, reconstructed_depth = unpatchify(pred, img_size=config['model']['image_size'], patch_size=config['model']['patch_size'])
        
        return pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched
    
def mae_loss(pred, target, mask, alpha=10.0, beta=1.0):
    """
    Calculates the MAE loss for a masked autoencoder, considering separate 
    losses for RGB and depth channels in the patch-based representation.

    Args:
        pred (torch.Tensor): Reconstructed patches from the decoder 
                            (shape: [batch_size, num_patches, patch_size*patch_size*n_channels])
        target (torch.Tensor): Original patches before masking 
                                (shape: [batch_size, num_patches, patch_size*patch_size*n_channels])
        mask (torch.Tensor): Binary mask indicating masked patches (0: keep, 1: masked) 
                                (shape: [batch_size, num_patches])
        alpha (float): Weighting factor for the RGB reconstruction loss.
        beta (float): Weighting factor for the depth reconstruction loss.

    Returns:
        tuple: A tuple containing:
            - loss (torch.Tensor): The total weighted MAE loss.
            - loss_rgb (torch.Tensor): The MAE loss for RGB channels.
            - loss_depth (torch.Tensor): The MAE loss for the depth channel.
    """
    print(f"MAE LOSS - pred shape: {pred.shape}")
    print(f"MAE LOSS - target shape: {target.shape}")
    print(f"MAE LOSS - mask shape: {mask.shape}")

    patch_size = config['model']['patch_size']
    num_channels_rgb = 3
    num_channels_depth = 1
    
    # Calculate the elements per patch for RGB and depth
    patch_elements_rgb = patch_size * patch_size * num_channels_rgb
    patch_elements_depth = patch_size * patch_size * num_channels_depth
    print(f"MAE LOSS - Patch elements RGB: {patch_elements_rgb}")
    print(f"MAE LOSS - Patch elements Depth: {patch_elements_depth}")
    
    assert target.shape[-1] == patch_elements_rgb + patch_elements_depth, f"Target should contain both RGB and depth information. Expected {patch_elements_rgb + patch_elements_depth}, got {target.shape[-1]}"
    #assert target.shape[-1] == patch_elements_rgb + patch_elements_depth, "Target should contain both RGB and depth information"

    # Extract RGB and depth predictions and targets from patches
    pred_rgb = pred[:, :, :patch_elements_rgb]
    pred_depth = pred[:, :, patch_elements_rgb:patch_elements_rgb + patch_elements_depth]
    print(f"MAE LOSS - pred_rgb shape: {pred_rgb.shape}")
    print(f"MAE LOSS - pred_depth shape: {pred_depth.shape}")
        
    # Decode the target to match the shape of the pred
    target_rgb = target[:, :, :patch_elements_rgb]
    target_depth = target[:, :, patch_elements_rgb:patch_elements_rgb + patch_elements_depth]
    print(f"MAE LOSS - target_rgb shape: {target_rgb.shape}")
    print(f"MAE LOSS - target_depth shape: {target_depth.shape}")
    print(f"MAE LOSS - target shape: {target.shape}")
    
    # Calculate mean squared error (MSE) for RGB and depth 
    loss_rgb = ((pred_rgb - target_rgb) ** 2).mean(dim=-1)  # Mean per-patch MSE for RGB
    loss_depth = ((pred_depth - target_depth) ** 2).mean(dim=-1)  # Mean per-patch MSE for depth
    print(f"MAE LOSS - Loss RGB shape: {loss_rgb.shape}")
    print(f"MAE LOSS - Loss Depth shape: {loss_depth.shape}")

    # Apply the mask to focus the loss on the masked patches
    loss_rgb = (loss_rgb * mask).sum() / mask.sum()  # Mean masked MSE for RGB
    loss_depth = (loss_depth * mask).sum() / mask.sum()  # Mean masked MSE for depth
    print(f"MAE LOSS - Loss RGB after mask: {loss_rgb}")
    print(f"MAE LOSS - Loss Depth after mask: {loss_depth}")

    # Calculate the total weighted loss
    loss = loss_rgb * alpha + loss_depth * beta
    print(f"MAE LOSS - Total loss: {loss}")
    print(f"MAE LOSS - Total loss shape: {loss.shape}")

    return loss, loss_rgb, loss_depth

if __name__ == "__main__":
    # Load configuration
    config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Mock example to test shapes
    batch_size = config['training']['batch_size']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']
    n_channels = config['model']['n_channels']  # RGB + depth
    d_model = config['model']['d_model']

    # Create random input tensor
    #x = torch.randn(batch_size, n_channels, img_size[0], img_size[1])
    
    #instead of creating a random tensor, load an image and depth map using the dataloader
    from data.dataloader import get_dataloaders
    train_loader, val_loader = get_dataloaders(config)
    for batch in train_loader:
        x = batch
        print(f"Batch shape: {x.shape}")
        print(f"Batch type: {type(x)}")
        print(f"Batch[0]: {x[0][:1]}")
        break
    

    # Initialize MAE model
    mae_model = MAE(d_model, img_size, patch_size, n_channels)

    # --- Forward Pass and Shape Checks ---
    print("=== Forward Pass ===")
    pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched = mae_model(x)
    

    print("\n=== Shape Verification ===")
    print(f"Input shape: {x.shape}")
    print(f"Prediction (Reconstructed Patches) shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"ids_restore shape: {ids_restore.shape}")
    print(f"Reconstructed Image shape: {reconstructed_image.shape}")
    print(f"Reconstructed Depth shape: {reconstructed_depth.shape}")

    # --- Loss Calculation (Optional) ---
    target = mae_model.encoder.patch_embed(x)
    target = mae_model.decoder.decoder_pred(target)  # This transforms the target to match pred's shape
    print(f"Target shape: {target.shape}")
    print(f"Prediction shape: {pred.shape}")

    # Ensure target_rgb and target_depth match the shapes of pred_rgb and pred_depth
    loss, loss_rgb, loss_depth = mae_loss(pred, target, mask)

    print(f"Loss: {loss.item():.4f}")
    print(f"RGB Loss: {loss_rgb.item():.4f}")
    print(f"Depth Loss: {loss_depth.item():.4f}")

    # --- Visualization ---
    #TODO: Add visualization code here
