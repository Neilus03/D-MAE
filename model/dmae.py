import torch
import torch.nn as nn
import numpy as np
import sys, os, yaml
import matplotlib.pyplot as plt
import wandb


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from model.vit import ViTFeatureExtractor, PatchEmbedding, TransformerEncoder, PositionalEncoding
from data.dataloader import denormalize_RGB, get_dataloaders

# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize WandB
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
        #print(f"Initialized RandomMasking with mask_ratio: {self.mask_ratio}")

    def forward(self, x):
        '''
        forward function:
        This function is used to randomly mask patches
        Args:
            x: Input patches (B, L, D)
        Returns:
            x_masked: Masked patches (B, L_visible, D)
            mask: Binary mask (B, L), where 1 indicates masked patches
            ids_restore: Indices to restore original order (B, L)
        '''
        N, L, D = x.shape  # Batch size, number of patches, dimension of each patch
        len_keep = int(L * (1 - self.mask_ratio))

        # Generate random noise for shuffling
        noise = torch.rand(N, L, device=x.device)
        # Sort noise to get the shuffling indices
        ids_shuffle = torch.argsort(noise, dim=1)
        # Restore indices to original order
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Select the patches to keep
        ids_keep = ids_shuffle[:, :len_keep]
        # Create the binary mask (1 for masked patches, 0 for visible patches)
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle the mask to match the original patch order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Mask the input patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        #print(f"RandomMasking - Input patches shape: {x.shape}")
        #print(f"RandomMasking - Mask shape: {mask.shape}")
        #print(f"RandomMasking - ids_restore shape: {ids_restore.shape}")
        #print(f"RandomMasking - Unmasked patches shape: {x_masked.shape}")

        return x_masked, mask, ids_restore


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
            x: Input visible patches (B, L_visible, D)
        Returns:
            x: Encoded patches (B, L_visible, D)
        '''
        #print(f"MAEEncoder - Input patches shape: {x.shape}")
        x = self.transformer_encoder(x)
        #print(f"MAEEncoder - After transformer encoder: {x.shape}")
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
        #print(f"MAEDecoder - Input encoded patches shape: {x.shape}")
        #print(f"MAEDecoder - ids_restore shape: {ids_restore.shape}")
        x = self.decoder_embed(x)
        #print(f"MAEDecoder - After embedding: {x.shape}")
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        #print(f"MAEDecoder - Mask tokens shape: {mask_tokens.shape}")
        x_ = torch.cat([x, mask_tokens], dim=1)
        #print(f"MAEDecoder - Concatenated shape: {x_.shape}")
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        #print(f"MAEDecoder - Gathered shape: {x_.shape}")

        for block in self.decoder_blocks:
            x_ = block(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        #print(f"MAEDecoder - Output (predicted patches) shape: {x_.shape}")

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
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # e.g., 14*14=196
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
            reconstructed_image: Reconstructed RGB image
            reconstructed_depth: Reconstructed depth map
            x_patched: Original patches before masking
        '''
        #print(f"MAE - Input image shape: {x.shape}")
        # Apply patch embedding
        x_patched = self.encoder.patch_embed(x)
        #print(f"MAE - After patch embedding: {x_patched.shape}")
        
        # Apply positional encoding BEFORE masking
        x_pos = self.encoder.positional_encoding(x_patched)
        #print(f"MAE - After positional encoding: {x_pos.shape}")
        
        # Apply masking to the patches
        x_masked, mask, ids_restore = self.random_masking(x_pos)
        
        # Encode the masked patches
        x_encoded = self.encoder(x_masked)
        
        # Decode patches
        pred = self.decoder(x_encoded, ids_restore)
        #print(f"MAE - Reconstructed patches shape: {pred.shape}")
        
        # Unpatchify the patches to reconstruct the image and depth map
        reconstructed_image, reconstructed_depth = unpatchify(pred, img_size=config['model']['image_size'], patch_size=config['model']['patch_size'])
        
        return pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched, x_masked


def mae_loss(pred, target, mask):
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
    #print(f"MAE LOSS - pred shape: {pred.shape}")
    #print(f"MAE LOSS - target shape: {target.shape}")
    #print(f"MAE LOSS - mask shape: {mask.shape}")

    patch_size = config['model']['patch_size']
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    num_channels_rgb = 3
    num_channels_depth = 1
    
    # Calculate the elements per patch for RGB and depth
    patch_elements_rgb = patch_size * patch_size * num_channels_rgb
    patch_elements_depth = patch_size * patch_size * num_channels_depth
    #print(f"MAE LOSS - Patch elements RGB: {patch_elements_rgb}")
    #print(f"MAE LOSS - Patch elements Depth: {patch_elements_depth}")
    
    assert target.shape[-1] == patch_elements_rgb + patch_elements_depth, f"Target should contain both RGB and depth information. Expected {patch_elements_rgb + patch_elements_depth}, got {target.shape[-1]}"
    
    # Extract RGB and depth predictions and targets from patches
    pred_rgb = pred[:, :, :patch_elements_rgb]
    pred_depth = pred[:, :, patch_elements_rgb:]
    #print(f"MAE LOSS - pred_rgb shape: {pred_rgb.shape}")
    #print(f"MAE LOSS - pred_depth shape: {pred_depth.shape}")
        
    # Extract target RGB and depth
    target_rgb = target[:, :, :patch_elements_rgb]
    target_depth = target[:, :, patch_elements_rgb:]
    #print(f"MAE LOSS - target_rgb shape: {target_rgb.shape}")
    #print(f"MAE LOSS - target_depth shape: {target_depth.shape}")
    
    # Calculate mean squared error (MSE) for RGB and depth 
    loss_rgb = ((pred_rgb - target_rgb) ** 2).mean(dim=-1)  # Mean per-patch MSE for RGB
    loss_depth = ((pred_depth - target_depth) ** 2).mean(dim=-1)  # Mean per-patch MSE for depth
    #print(f"MAE LOSS - Loss RGB shape: {loss_rgb.shape}")
    #print(f"MAE LOSS - Loss Depth shape: {loss_depth.shape}")

    # Apply the mask to focus the loss on the masked patches
    loss_rgb = (loss_rgb * mask).sum() / mask.sum()  # Mean masked MSE for RGB
    loss_depth = (loss_depth * mask).sum() / mask.sum()  # Mean masked MSE for depth
    #print(f"MAE LOSS - Loss RGB after mask: {loss_rgb}")
    #print(f"MAE LOSS - Loss Depth after mask: {loss_depth}")

    # Calculate the total weighted loss
    loss = loss_rgb * alpha + loss_depth * beta
    #print(f"MAE LOSS - Total loss: {loss}")
    #print(f"MAE LOSS - Total loss shape: {loss.shape}")

    return loss, loss_rgb, loss_depth


def unpatchify(patches, img_size=(224, 224), patch_size=16):
    '''
    Reconstructs the image from patches.

    Args:
        patches (torch.Tensor): The output patches from the decoder
                                (shape: [batch_size, num_patches, patch_size*patch_size*num_channels]).
        img_size (tuple): The original image size (height, width).
        patch_size (int): The size of each patch (assuming square patches).

    Returns:
        tuple: (reconstructed_rgb, reconstructed_depth)
            reconstructed_rgb: [batch_size, 3, height, width]
            reconstructed_depth: [batch_size, 1, height, width]
    '''
    batch_size, num_patches, patch_elements = patches.shape
    num_channels = patch_elements // (patch_size * patch_size)
    assert num_channels == 4, "Expected 4 channels (RGB + Depth)"
    assert patch_elements == patch_size * patch_size * num_channels, "Incorrect patch elements"

    # Calculate the number of patches per dimension
    num_patches_per_dim = img_size[0] // patch_size
    assert num_patches_per_dim ** 2 == num_patches, "Number of patches does not match image dimensions"

    # Reshape patches into the image grid
    patches = patches.view(batch_size, num_patches_per_dim, num_patches_per_dim, patch_size, patch_size, num_channels)
    # Rearrange dimensions to match image shape
    patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    # Combine patches into full images
    images = patches.view(batch_size, num_channels, img_size[0], img_size[1])

    # Split the channels into RGB and depth
    reconstructed_rgb = images[:, :3, :, :]
    reconstructed_depth = images[:, 3:, :, :]

    #print(f"Unpatchify - Reconstructed RGB image shape: {reconstructed_rgb.shape}")
    #print(f"Unpatchify - Reconstructed Depth image shape: {reconstructed_depth.shape}")

    return reconstructed_rgb, reconstructed_depth


def extract_patches(image, patch_size):
    '''
    Extract patches from an image tensor.
    Args:
        image: Tensor of shape (C, H, W)
        patch_size: size of the patches
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size)
    '''
    # Unfold the image to get patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Rearrange dimensions to get patches in (num_patches, channels, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image.shape[0], patch_size, patch_size)
    return patches

def assemble_patches_with_gaps(patches, gap_size, num_patches_per_row, patch_size, num_channels=3, depth=False):
    '''
    Assembles patches into an image with gaps between patches.
    Args:
        patches: numpy array of shape (num_patches, num_channels, patch_size, patch_size)
        gap_size: size of the gap between patches (in pixels)
        num_patches_per_row: number of patches per row/column
        patch_size: size of each patch (assuming square patches)
        num_channels: number of channels in the image
        depth: whether it's a depth map (True) or RGB image (False)
    Returns:
        image_with_gaps: numpy array of shape (grid_size, grid_size, num_channels) or (grid_size, grid_size)
    '''
    grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size
    if depth:
        image_with_gaps = np.ones((grid_size, grid_size))
    else:
        image_with_gaps = np.ones((grid_size, grid_size, num_channels))
    idx = 0
    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            y_start = row * (patch_size + gap_size)
            x_start = col * (patch_size + gap_size)
            if depth:
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches[idx]
            else:
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx].transpose(1, 2, 0)
            idx += 1
    return image_with_gaps


if __name__ == "__main__":
    # Load configuration
    config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load data using the updated dataloader
    train_loader, val_loader = get_dataloaders(config)
    for batch in train_loader:
        image_depths, patches = batch  # Unpack the batch
        x = image_depths
        #print(f"Batch image_depths shape: {image_depths.shape}")
        #print(f"Batch patches shape: {patches.shape}")
        break

    # Initialize MAE model
    d_model = config['model']['d_model']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']
    n_channels = config['model']['n_channels']  # RGB + depth

    mae_model = MAE(d_model, img_size, patch_size, n_channels)

    # --- Forward Pass and Shape Checks ---
    #print("=== Forward Pass ===")
    pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched, x_masked = mae_model(x)

    #print("\n=== Shape Verification ===")
    #print(f"Input shape: {x.shape}")
    #print(f"Prediction (Reconstructed Patches) shape: {pred.shape}")
    #print(f"Mask shape: {mask.shape}")
    #print(f"ids_restore shape: {ids_restore.shape}")
    #print(f"Reconstructed Image shape: {reconstructed_image.shape}")
    #print(f"Reconstructed Depth shape: {reconstructed_depth.shape}")

    # --- Loss Calculation ---
    # Pass x_patched through the decoder's prediction head to match pred's shape
    target = mae_model.decoder.decoder_pred(x_patched)
    #print(f"Target shape: {target.shape}")
    #print(f"Prediction shape: {pred.shape}")

    # Compute loss
    loss, loss_rgb, loss_depth = mae_loss(pred, target, mask)

    print(f"Loss: {loss.item():.4f}")
    print(f"RGB Loss: {loss_rgb.item():.4f}")
    print(f"Depth Loss: {loss_depth.item():.4f}")

    # --- Visualizations ---
    # Denormalization parameters
    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']

    # Select the first sample in the batch
    original_image_depth = x[0]  # Shape: (4, H, W)

    # Denormalize original RGB image
    original_rgb_image = denormalize_RGB(original_image_depth[:3]).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    # Denormalize original depth map
    original_depth_map = original_image_depth[3].cpu().numpy()
    original_depth_map = original_depth_map * depth_std + depth_mean

    # Save original RGB image and depth map
    plt.imsave('original_rgb_image.png', original_rgb_image)
    plt.imsave('original_depth_map.png', original_depth_map, cmap='viridis')

    # Extract patches from the original image
    original_patches = extract_patches(original_image_depth, patch_size)  # Shape: (num_patches, 4, patch_size, patch_size)
    original_patches_rgb = original_patches[:, :3, :, :]
    original_patches_depth = original_patches[:, 3:, :, :]

    # Denormalize patches
    original_patches_rgb_denorm = denormalize_RGB(original_patches_rgb).clamp(0, 1).cpu().numpy()  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth.cpu().numpy()
    original_patches_depth_denorm = original_patches_depth_denorm * depth_std + depth_mean  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)

    # Assemble original patches with gaps
    gap_size = 2  # pixels
    num_patches_per_row = img_size[0] // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_denorm, gap_size, num_patches_per_row, patch_size, depth=True)

    # Save assembled original patches
    plt.imsave('assembled_original_rgb_patches.png', assembled_original_rgb)
    plt.imsave('assembled_original_depth_patches.png', assembled_original_depth, cmap='viridis')

    # Create masked patches
    masked_patches = original_patches.clone()
    masked_indices = (mask[0] == 1).nonzero(as_tuple=False).squeeze()
    masked_patches[masked_indices] = 0  # Set masked patches to zero

    masked_patches_rgb = masked_patches[:, :3, :, :]
    masked_patches_depth = masked_patches[:, 3:, :, :]

    # Denormalize masked patches
    masked_patches_rgb_denorm = denormalize_RGB(masked_patches_rgb).clamp(0, 1).cpu().numpy()
    masked_patches_depth_denorm = masked_patches_depth.cpu().numpy()
    masked_patches_depth_denorm = masked_patches_depth_denorm * depth_std + depth_mean
    masked_patches_depth_denorm = masked_patches_depth_denorm.squeeze(1)

    # Assemble masked patches with gaps
    assembled_masked_rgb = assemble_patches_with_gaps(masked_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_masked_depth = assemble_patches_with_gaps(masked_patches_depth_denorm, gap_size, num_patches_per_row, patch_size, depth=True)

    # Save assembled masked patches
    plt.imsave('assembled_masked_rgb_patches.png', assembled_masked_rgb)
    plt.imsave('assembled_masked_depth_patches.png', assembled_masked_depth, cmap='viridis')

    # Denormalize reconstructed RGB image
    reconstructed_rgb_denorm = denormalize_RGB(reconstructed_image[0]).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    # Denormalize reconstructed depth map
    reconstructed_depth_map = reconstructed_depth[0, 0].detach().cpu().numpy()
    reconstructed_depth_map = reconstructed_depth_map * depth_std + depth_mean

    # Save reconstructed RGB image and depth map
    plt.imsave('reconstructed_rgb_image.png', reconstructed_rgb_denorm)
    plt.imsave('reconstructed_depth_map.png', reconstructed_depth_map, cmap='viridis')

    # Log images to WandB
    wandb.log({
        "Original RGB Image": wandb.Image('original_rgb_image.png'),
        "Original Depth Map": wandb.Image('original_depth_map.png'),
        "Assembled Original RGB Patches": wandb.Image('assembled_original_rgb_patches.png'),
        "Assembled Original Depth Patches": wandb.Image('assembled_original_depth_patches.png'),
        "Masked RGB Image": wandb.Image('assembled_masked_rgb_patches.png'),
        "Masked Depth Map": wandb.Image('assembled_masked_depth_patches.png'),
        "Reconstructed RGB Image": wandb.Image('reconstructed_rgb_image.png'),
        "Reconstructed Depth Map": wandb.Image('reconstructed_depth_map.png')
    })

    # Close WandB run
    wandb.finish()

