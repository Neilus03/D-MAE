# Description: Utility functions for visualizing images and depth maps during training and validation.

import numpy as np
import sys, os, yaml
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data.dataloader import denormalize_RGB, denormalize_depth


# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

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

    print(f"Unpatchify - Reconstructed RGB image shape: {reconstructed_rgb.shape}")
    print(f"Unpatchify - Reconstructed Depth image shape: {reconstructed_depth.shape}")

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
    Assembles patches into an image with gaps between patches for cooler visualization.
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
    #define the grid size as the size of the image with gaps
    grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size # in pixels
    if depth:
        #for depth use a single channel
        image_with_gaps = np.ones((grid_size, grid_size))
    else:
        #for RGB use 3 channels
        image_with_gaps = np.ones((grid_size, grid_size, num_channels))
    idx = 0
    #iterate over the patches and place them in the image with gaps
    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            #calculate the start position of the patch
            y_start = row * (patch_size + gap_size)
            x_start = col * (patch_size + gap_size)
            if depth:
                #for depth maps we only have one channel
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches[idx]
            else:
                #for RGB images we have 3 channels
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx].transpose(1, 2, 0)
            idx += 1
    return image_with_gaps

def log_visualizations(image_depths, reconstructed_image, reconstructed_depth, mask, epoch, batch_idx, prefix='Train'):
    '''
    Logs visualizations to WandB.
    Args:
        image_depths: input images with depth (batch_size, 4, H, W)
        reconstructed_image: reconstructed RGB images (batch_size, 3, H, W)
        reconstructed_depth: reconstructed depth maps (batch_size, 1, H, W)
        mask: mask used during training (batch_size, num_patches)
        epoch: current epoch
        batch_idx: current batch index
        prefix: 'Train' or 'Validation'
    '''
    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']

    # Select the first sample in the batch
    original_image_depth = image_depths[0].detach().cpu()  # Shape: (4, H, W)

    # Denormalize original RGB image
    original_rgb_image = denormalize_RGB(original_image_depth[:3]).permute(1, 2, 0).clamp(0, 1).numpy()
    # Denormalize original depth map
    original_depth_map = original_image_depth[3].numpy()
    original_depth_map = original_depth_map * depth_std + depth_mean

    # Normalize depth map for visualization
    original_depth_map_viz = (original_depth_map - original_depth_map.min()) / (original_depth_map.max() - original_depth_map.min() + 1e-8)

    # Extract patches from the original image
    original_patches = extract_patches(original_image_depth, patch_size)  # Shape: (num_patches, 4, patch_size, patch_size)
    original_patches_rgb = original_patches[:, :3, :, :]
    original_patches_depth = original_patches[:, 3:, :, :]

    # Denormalize patches
    original_patches_rgb_denorm = denormalize_RGB(original_patches_rgb).clamp(0, 1).numpy()  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth.numpy()
    original_patches_depth_denorm = original_patches_depth_denorm * depth_std + depth_mean  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)

    # Normalize depth patches for visualization
    depth_min = original_patches_depth_denorm.min()
    depth_max = original_patches_depth_denorm.max()
    original_patches_depth_viz = (original_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble original patches with gaps
    gap_size = 2  # pixels
    num_patches_per_row = img_size[0] // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Masked patches
    masked_patches = original_patches.clone()
    masked_indices = (mask[0] == 1).nonzero(as_tuple=False).squeeze()
    masked_patches[masked_indices] = 0  # Set masked patches to zero

    # Extract masked patches for rgb and depth separately
    masked_patches_rgb = masked_patches[:, :3, :, :] # Shape: (num_patches, 3, patch_size, patch_size)
    masked_patches_depth = masked_patches[:, 3:, :, :] # Shape: (num_patches, 1, patch_size, patch_size)

    # Denormalize masked patches
    masked_patches_rgb_denorm = denormalize_RGB(masked_patches_rgb).clamp(0, 1).numpy()
    masked_patches_depth_denorm = masked_patches_depth.numpy()
    # Denormalize masked depth patches
    masked_patches_depth_denorm = denormalize_depth(masked_patches_depth, depth_mean, depth_std)
    masked_patches_depth_denorm = masked_patches_depth_denorm.squeeze(1)

    # Normalize masked depth patches for visualization
    depth_min = masked_patches_depth_denorm.min()
    depth_max = masked_patches_depth_denorm.max()
    masked_patches_depth_viz = (masked_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble masked patches with gaps
    assembled_masked_rgb = assemble_patches_with_gaps(masked_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_masked_depth = assemble_patches_with_gaps(masked_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Reconstructed images
    reconstructed_rgb_denorm = denormalize_RGB(reconstructed_image[0].detach().cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
    reconstructed_depth_map = reconstructed_depth[0, 0].detach().cpu().numpy()
    reconstructed_depth_map = reconstructed_depth_map * depth_std + depth_mean

    # Normalize reconstructed depth map for visualization
    recon_depth_min = reconstructed_depth_map.min()
    recon_depth_max = reconstructed_depth_map.max()
    reconstructed_depth_map_viz = (reconstructed_depth_map - recon_depth_min) / (recon_depth_max - recon_depth_min + 1e-8)

    # Create depth images using matplotlib and save them to buffers
    depth_images = {}
    # Original Depth Map
    fig1 = plt.figure()
    plt.imshow(original_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    buf1.seek(0)
    depth_images[f'{prefix} Original Depth Map'] = wandb.Image(Image.open(buf1), caption='Original Depth Map')

    # Assembled Original Depth Patches
    fig2 = plt.figure()
    plt.imshow(assembled_original_depth, cmap='viridis')
    plt.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    buf2.seek(0)
    depth_images[f'{prefix} Assembled Original Depth Patches'] = wandb.Image(Image.open(buf2), caption='Original Depth Patches')

    # Masked Depth Map
    fig3 = plt.figure()
    plt.imshow(assembled_masked_depth, cmap='viridis')
    plt.axis('off')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    buf3.seek(0)
    depth_images[f'{prefix} Masked Depth Map'] = wandb.Image(Image.open(buf3), caption='Masked Depth Map')

    # Reconstructed Depth Map
    fig4 = plt.figure()
    plt.imshow(reconstructed_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig4)
    buf4.seek(0)
    depth_images[f'{prefix} Reconstructed Depth Map'] = wandb.Image(Image.open(buf4), caption='Reconstructed Depth Map')

    # Log images to WandB
    wandb.log({
        f'{prefix} Original RGB Image': wandb.Image(original_rgb_image),
        f'{prefix} Assembled Original RGB Patches': wandb.Image(assembled_original_rgb),
        f'{prefix} Masked RGB Image': wandb.Image(assembled_masked_rgb),
        f'{prefix} Reconstructed RGB Image': wandb.Image(reconstructed_rgb_denorm),
        **depth_images,
        'Epoch': epoch,
        'Batch': batch_idx
    })