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
        image_depth = torch.cat((image, depth), dim=0)

        return image_depth

def get_dataloaders(config):
    image_size = tuple(config['model']['image_size'])
    batch_size = config['training']['batch_size']
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']

    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_mean= config['data']['depth_stats']['mean']
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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


###########################################################################
# ANALYZE THE MEAN AND STD OF THE DEPTH MAPS TO NOW HOW TO NORMALIZE THEM #
###########################################################################

def analyze_depth_stats(loader):
    depth_min, depth_max = float('inf'), float('-inf')
    depth_sum, depth_sq_sum, count = 0, 0, 0

    for batch in loader:
        depth = batch[:, 3]  # Assuming depth is the 4th channel
        depth_min = min(depth_min, depth.min().item())
        depth_max = max(depth_max, depth.max().item())
        depth_sum += depth.sum().item()
        depth_sq_sum += (depth ** 2).sum().item()
        count += depth.numel()

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
    for i, batch in enumerate(train_loader):
        logger.info(f"Batch {i+1}:")
        logger.info(f"  Shape: {batch.shape}")
        logger.info(f"  Data type: {batch.dtype}")
        logger.info(f"  RGB value range: [{batch[:, :3].min():.4f}, {batch[:, :3].max():.4f}]")
        logger.info(f"  Depth value range: [{batch[:, 3].min():.4f}, {batch[:, 3].max():.4f}]")
        
        if i == 0:
            # Visualize first image and depth map
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(denormalize_RGB(batch[0, :3]).permute(1, 2, 0).clamp(0, 1).numpy())
            plt.title("RGB Image")
            plt.subplot(122)
            plt.imshow(batch[0, 3].numpy(), cmap='viridis')
            plt.title("Depth Map")
            plt.savefig('sample_data_visualization.png')
            plt.close()

        if i == 4:  # Check first 5 batches
            break

    logger.info("Calculating depth statistics for entire dataset...")
    depth_mean, depth_std = analyze_depth_stats(train_loader)

    logger.info("Data loading and analysis complete.")
