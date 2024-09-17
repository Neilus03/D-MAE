import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Error loading config file: {e}")
    raise

class CustomDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform_image=None, transform_depth=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        self.transform_image = transform_image
        self.transform_depth = transform_depth

        # Verify file count match
        if len(self.image_files) != len(self.depth_files):
            raise ValueError(f"Mismatch in number of images ({len(self.image_files)}) and depth maps ({len(self.depth_files)})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

        try:
            depth = np.load(depth_path)
            depth = Image.fromarray(depth.astype(np.float32))
        except Exception as e:
            logger.error(f"Error loading depth map {depth_path}: {e}")
            raise

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        # Concatenate the depth map as the 4th channel
        image_depth = torch.cat((image, depth), dim=0)  # shape: (4, H, W)

        # Extract patches from the image_depth tensor
        patch_size = 16  # Each patch is 16x16
        # Ensure the image dimensions are divisible by the patch size
        _, H, W = image_depth.shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Image dimensions ({H}, {W}) are not divisible by patch size ({patch_size})")

        # Unfold the image to get patches
        patches = image_depth.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # Rearrange dimensions to get patches in (num_patches, channels, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 4, patch_size, patch_size)  # shape: (196, 4, 16, 16)

        return image_depth, patches  # shape: (4, H, W), (196, 4, 16, 16)

def get_dataloaders(config):
    '''
    Creates PyTorch DataLoader objects for the training and validation datasets
    
    Args:
        config: Dictionary containing configuration parameters
    
    Returns:
        train_loader: DataLoader object for the training dataset
        val_loader: DataLoader object for the validation dataset
    '''
    image_size = tuple(config['model']['image_size'])
    batch_size = config['training']['batch_size']
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']

    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    transform_depth = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[depth_mean], std=[depth_std])
    ])
    
    train_image_dir = os.path.join(train_dir, 'images')
    train_depth_dir = os.path.join(train_dir, 'depth')
    val_image_dir = os.path.join(val_dir, 'images')
    val_depth_dir = os.path.join(val_dir, 'depth')

    train_dataset = CustomDataset(train_image_dir, train_depth_dir, transform_image=transform_image, transform_depth=transform_depth)
    val_dataset = CustomDataset(val_image_dir, val_depth_dir, transform_image=transform_image, transform_depth=transform_depth)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def denormalize_RGB(tensor):
    '''
    Denormalizes the RGB channels of a tensor containing an RGB or RGB-D image
    '''
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    
    # Check if the input is a 4D tensor (batch, channels, height, width)
    if tensor.dim() == 4:
        # Only denormalize the first 3 channels (RGB)
        denormalized_rgb = tensor[:, :3] * std + mean
        
        # If there's a depth channel, keep it unchanged
        if tensor.shape[1] == 4:
            denormalized_tensor = torch.cat([denormalized_rgb, tensor[:, 3:]], dim=1)
        else:
            denormalized_tensor = denormalized_rgb
    else:
        # For 3D tensors (single image)
        if tensor.shape[0] >= 3:
            denormalized_rgb = tensor[:3] * std + mean
            if tensor.shape[0] == 4:
                denormalized_tensor = torch.cat([denormalized_rgb, tensor[3:]], dim=0)
            else:
                denormalized_tensor = denormalized_rgb
        else:
            denormalized_tensor = tensor * std + mean

    return denormalized_tensor

def denormalize_depth(tensor, mean, std):
    '''
    Denormalizes the depth channel of a tensor containing an RGB-D image
    '''
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    denormalized_depth = tensor * std + mean
    return denormalized_depth

def analyze_depth_stats(loader):
    depth_min, depth_max = float('inf'), float('-inf')
    depth_sum, depth_sq_sum, count = 0, 0, 0

    for image_depths, _ in loader:
        depth = image_depths[:, 3]  # Assuming depth is the 4th channel
        depth_min = min(depth_min, depth.min().item())
        depth_max = max(depth_max, depth.max().item())
        depth_sum += depth.sum().item()
        depth_sq_sum += (depth ** 2).sum().item()
        count += depth.numel() / depth.shape[1]  # Adjust for batch size

    depth_mean = depth_sum / count
    depth_std = (depth_sq_sum / count - depth_mean ** 2) ** 0.5

    logger.info(f"Depth statistics:")
    logger.info(f"  Min: {depth_min:.4f}")
    logger.info(f"  Max: {depth_max:.4f}")
    logger.info(f"  Mean: {depth_mean:.4f}")
    logger.info(f"  Std: {depth_std:.4f}")

    return depth_mean, depth_std

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(config)

    logger.info("Analyzing training data:")
    for i, (image_depths, patches) in enumerate(train_loader):
        logger.info(f"Batch {i+1}:")
        logger.info(f"  image_depths Shape: {image_depths.shape}")  # Shape: (batch_size, 4, H, W)
        logger.info(f"  patches Shape: {patches.shape}")  # Shape: (batch_size, 196, 4, 16, 16)
        logger.info(f"  Data type: {image_depths.dtype}")
        logger.info(f"  RGB value range: [{image_depths[:, :3].min():.4f}, {image_depths[:, :3].max():.4f}]")
        logger.info(f"  Depth value range: [{image_depths[:, 3].min():.4f}, {image_depths[:, 3].max():.4f}]")
        
        if i == 0:
            # Visualize and save the first image and depth map
            import matplotlib.pyplot as plt
            import os
            import numpy as np

            # Create a directory to save images
            os.makedirs('sample_images', exist_ok=True)

            # Denormalize RGB image
            rgb_image = denormalize_RGB(image_depths[0, :3]).permute(1, 2, 0).clamp(0, 1).numpy()
            # Denormalize depth map
            depth_mean = config['data']['depth_stats']['mean']
            depth_std = config['data']['depth_stats']['std']
            depth_map = image_depths[0, 3].numpy()
            depth_map = depth_map * depth_std + depth_mean

            # Save RGB image
            plt.imsave('sample_images/original_rgb_image.png', rgb_image)
            # Save depth map
            plt.imsave('sample_images/original_depth_map.png', depth_map, cmap='viridis')

            # Save first few patches
            num_patches_to_save = 5
            for j in range(num_patches_to_save):
                patch = patches[0, j]  # First image in batch, patch j
                # Denormalize RGB channels of the patch
                patch_rgb = denormalize_RGB(patch[:3]).permute(1, 2, 0).clamp(0, 1).numpy()
                # Depth channel of the patch
                patch_depth = patch[3].numpy()
                # Denormalize depth
                patch_depth = patch_depth * depth_std + depth_mean

                # Save RGB patch
                plt.imsave(f'sample_images/patch_{j}_rgb.png', patch_rgb)
                # Save depth patch
                plt.imsave(f'sample_images/patch_{j}_depth.png', patch_depth, cmap='viridis')

            # Reconstruct image from patches with gaps between patches
            # Define gap size
            gap_size = 2  # pixels
            num_patches_per_row = 14
            patch_size = 16
            grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size

            # Initialize empty arrays with gaps
            reconstructed_rgb_with_gaps = np.ones((grid_size, grid_size, 3))
            reconstructed_depth_with_gaps = np.ones((grid_size, grid_size))

            # Get denormalized patches
            patches_rgb = denormalize_RGB(patches[0, :, :3]).clamp(0, 1).numpy()  # Shape: (196, 3, 16, 16)
            patches_depth = patches[0, :, 3].numpy()  # Shape: (196, 16, 16)
            patches_depth = patches_depth * depth_std + depth_mean

            # Fill the reconstructed images with patches and gaps
            idx = 0
            for row in range(num_patches_per_row):
                for col in range(num_patches_per_row):
                    y_start = row * (patch_size + gap_size)
                    x_start = col * (patch_size + gap_size)
                    reconstructed_rgb_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches_rgb[idx].transpose(1, 2, 0)
                    reconstructed_depth_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches_depth[idx]
                    idx += 1

            # Save reconstructed images with gaps
            plt.imsave('sample_images/reconstructed_rgb_with_gaps.png', reconstructed_rgb_with_gaps)
            plt.imsave('sample_images/reconstructed_depth_with_gaps.png', reconstructed_depth_with_gaps, cmap='viridis')

            # Also save the reconstructed image without gaps for comparison
            # Reshape patches into grid
            patches_image = patches[0]  # Shape: (196, 4, 16, 16)
            # Reshape to (14,14,4,16,16)
            patches_image = patches_image.view(14, 14, 4, 16, 16)
            # Permute to (4, 14, 16, 14, 16)
            patches_image = patches_image.permute(2, 0, 3, 1, 4)
            # Reshape to (4, H, W)
            reconstructed_image = patches_image.contiguous().view(4, 14*16, 14*16)

            # Denormalize and save reconstructed RGB image
            reconstructed_rgb_image = denormalize_RGB(reconstructed_image[:3]).permute(1, 2, 0).clamp(0, 1).numpy()
            plt.imsave('sample_images/reconstructed_rgb_image.png', reconstructed_rgb_image)
            # Reconstructed depth map
            reconstructed_depth_map = reconstructed_image[3].numpy()
            # Denormalize depth
            reconstructed_depth_map = reconstructed_depth_map * depth_std + depth_mean
            plt.imsave('sample_images/reconstructed_depth_map.png', reconstructed_depth_map, cmap='viridis')

        if i == 4:  # Check first 5 batches
            break

    logger.info("Calculating depth statistics for entire dataset...")
    #depth_mean, depth_std = analyze_depth_stats(train_loader)

    logger.info("Data loading and analysis complete.")
